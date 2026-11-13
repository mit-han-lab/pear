// The RoboDojo Apps' inotify (RoboDojoApps in apps.py builds and preloads it): it never reports a change, as nothing an
// App watches changes during its episode. Each instance is a pipe that only inotify_rm_watch writes to, so an App holds
// no kernel instance: the kernel limits a user's per machine, and omniclient's file provider crashes or hangs without
// one.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_FD 65536  // instances whose read end would be above this are refused (EMFILE)

static int writers[MAX_FD];  // an instance's pipe write end + 1, by its read end; 0: not one of these instances
static int next_wd = 1;      // the watch descriptors handed out, positive as the kernel's

static int fake(int fd) {
    return fd >= 0 && fd < MAX_FD && writers[fd] != 0;
}

int inotify_init1(int flags) {
    int fds[2];
    if (flags & ~(IN_CLOEXEC | IN_NONBLOCK)) {
        errno = EINVAL;
        return -1;
    }
    if (pipe2(fds, flags) < 0)  // IN_CLOEXEC and IN_NONBLOCK are O_CLOEXEC and O_NONBLOCK
        return -1;
    if (fds[0] >= MAX_FD) {
        close(fds[0]);
        close(fds[1]);
        errno = EMFILE;
        return -1;
    }
    // the write end: never inherited, and never blocking (a reader that stopped reading loses IN_IGNORED events)
    fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    fcntl(fds[1], F_SETFL, O_NONBLOCK);
    __atomic_store_n(&writers[fds[0]], fds[1] + 1, __ATOMIC_RELEASE);
    return fds[0];
}

int inotify_init(void) {
    return inotify_init1(0);
}

int inotify_add_watch(int fd, const char *path, uint32_t mask) {
    static int (*real)(int, const char *, uint32_t);
    struct stat st;
    if (!fake(fd)) {
        if (!real)
            real = (int (*)(int, const char *, uint32_t))dlsym(RTLD_NEXT, "inotify_add_watch");
        return real(fd, path, mask);
    }
    // the kernel's answers about the path itself: a missing path or a non-directory under IN_ONLYDIR fails as there
    if ((mask & IN_DONT_FOLLOW ? lstat(path, &st) : stat(path, &st)) < 0)
        return -1;
    if ((mask & IN_ONLYDIR) && !S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        return -1;
    }
    return __atomic_fetch_add(&next_wd, 1, __ATOMIC_RELAXED);
}

int inotify_rm_watch(int fd, int wd) {
    static int (*real)(int, int);
    struct inotify_event event = {.wd = wd, .mask = IN_IGNORED};
    if (!fake(fd)) {
        if (!real)
            real = (int (*)(int, int))dlsym(RTLD_NEXT, "inotify_rm_watch");
        return real(fd, wd);
    }
    // a full pipe drops the event (its reader stopped reading), as a kernel queue overflow would
    if (write(__atomic_load_n(&writers[fd], __ATOMIC_ACQUIRE) - 1, &event, sizeof event) < 0)
        return 0;
    return 0;
}
