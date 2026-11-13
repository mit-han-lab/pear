# Ray gives the head node one unit of the resource "node:__internal_head__" (ray._common.constants): an actor that
# asks for a sliver of it runs there, as Ray Train's state actor does. Without it, an actor that asks for no CPU may
# land on a simulator node (which offers none) and end with that node's job.
ON_RAY_HEAD = {"node:__internal_head__": 0.001}
