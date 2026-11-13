import asyncio
from uuid import uuid4

from aiohttp import web

from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server.command_cache import CommandCache
from pear.deploy.station.gui.server.service_identity import ServiceIdentity
from pear.deploy.station.records.image_uploads import ImageUploads


class ConsoleRoutes:
    """The service's identity, the console's state and model check, the browsers' claims of control, the commands and
    the task images."""

    UPLOAD_SLOTS = 2  # uploads read and stored at once
    UPLOAD_TIMEOUT_S = 60  # for one upload to be read and stored
    UPLOAD_OVERHEAD_BYTES = 65536  # multipart framing around the image bytes
    CHUNK_BYTES = 65536  # of an upload, read at a time

    def __init__(self, console: Console, command_cache: CommandCache, identity: ServiceIdentity) -> None:
        self.console = console
        self.command_cache = command_cache
        self.identity = identity  # who the service is, which /health answers
        self.instance_id = uuid4().hex  # of this run of the service: the page starts over when it changes
        self.upload_slots = asyncio.Semaphore(self.UPLOAD_SLOTS)

    def add_routes(self, router: web.UrlDispatcher) -> None:
        """Adds the routes to ``router``."""
        router.add_get(ServiceIdentity.HEALTH_PATH, self.health)
        router.add_get("/api/state", self.state)
        router.add_get("/api/model/health", self.model_health)
        router.add_post("/api/control/claim", self.claim)
        router.add_post("/api/control/heartbeat", self.heartbeat)
        router.add_post("/api/command", self.command)
        router.add_post("/api/images", self.upload_image)
        router.add_delete("/api/images/{image_id}", self.delete_image)

    async def health(self, request: web.Request) -> web.Response:
        """Answers who the service is."""
        return web.json_response(self.identity.health(self.instance_id))

    async def state(self, request: web.Request) -> web.Response:
        """Answers the console's state."""
        return web.json_response(await asyncio.to_thread(self.console.snapshot))

    async def model_health(self, request: web.Request) -> web.Response:
        """Answers the selected model's connection, checked anew."""
        # the operator's Check connection: a new check, unless one ended while this request waited for it
        return web.json_response(await asyncio.to_thread(self.console.check_model, is_forced=True))

    async def claim(self, request: web.Request) -> web.Response:
        """Gives control to the browser unless another holds it, and answers whether the browser holds it."""
        client_id = await self.read_client_id(request)
        return web.json_response({"owner": self.console.lease.claim(client_id)})

    async def heartbeat(self, request: web.Request) -> web.Response:
        """Renews the browser's control, and answers whether the browser holds it."""
        # browsers send their heartbeats over /api/live; scripted clients use this route
        client_id = await self.read_client_id(request)
        return web.json_response({"owner": self.console.lease.heartbeat(client_id)})

    async def command(self, request: web.Request) -> web.Response:
        """Answers a command once it ran."""
        answer, status = await self.command_cache.run(await request.json())
        return web.json_response(answer, status=status)

    async def upload_image(self, request: web.Request) -> web.Response:
        """Stores one draft image for the next task of the browser that holds control."""
        client_id = request.query.get("client_id")
        self.console.lease.require(client_id)
        if self.console.task_run.task:
            raise ValueError("Clear the current task before attaching images")
        if not self.console.supports_reference_images():
            raise ValueError("This model service does not yet support task images")
        if request.content_type != "multipart/form-data":
            raise ValueError("Upload one image using multipart/form-data")

        limit = ImageUploads.MAX_BYTES + self.UPLOAD_OVERHEAD_BYTES
        request = request.clone(client_max_size=limit)
        if request.content_length and request.content_length > limit:
            raise web.HTTPRequestEntityTooLarge(max_size=ImageUploads.MAX_BYTES, actual_size=request.content_length)

        async with self.upload_slots, asyncio.timeout(self.UPLOAD_TIMEOUT_S):
            filename, image_bytes = await self.read_image(request)
            self.console.lease.require(client_id)
            draft_image = await asyncio.to_thread(self.console.uploads.add, client_id, filename, image_bytes)
        return web.json_response(draft_image)

    async def delete_image(self, request: web.Request) -> web.Response:
        """Deletes a draft image of the browser that holds control."""
        client_id = request.query.get("client_id")
        self.console.lease.require(client_id)
        await asyncio.to_thread(self.console.uploads.delete, request.match_info["image_id"], client_id)

        return web.json_response({"deleted": True})

    async def read_client_id(self, request: web.Request) -> str:
        """Returns the browser's id the request's body gives, and refuses one of another form."""
        body = await request.json()
        client_id = body.get("client_id")
        if not isinstance(client_id, str) or not CommandCache.ID_PATTERN.fullmatch(client_id):
            raise ValueError("Invalid browser ID")

        return client_id

    async def read_image(self, request: web.Request) -> tuple[str, bytearray]:
        """Returns the file name and the bytes of the one image the multipart request carries."""
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "image" or not part.filename:
            raise ValueError("Upload one image file")

        image_bytes = bytearray()
        while chunk := await part.read_chunk(self.CHUNK_BYTES):
            image_bytes.extend(chunk)
            if len(image_bytes) > ImageUploads.MAX_BYTES:
                raise web.HTTPRequestEntityTooLarge(max_size=ImageUploads.MAX_BYTES, actual_size=len(image_bytes))
        if await reader.next() is not None:
            raise ValueError("Upload one image at a time")

        return part.filename, image_bytes
