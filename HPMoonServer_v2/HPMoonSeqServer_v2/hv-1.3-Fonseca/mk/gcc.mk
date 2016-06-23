# -*- Makefile-gmake -*-
GCC_VERSION := $(shell echo `gcc -dumpversion | cut -f1-2 -d.` \>= 4.3 | bc )

gcc-guess-march = $(strip $(shell ${CC} -march=$(MARCH) -x c -S -\#\#\# - < /dev/null 2>&1 | \
                grep -m 1 -o -e "march=[^'\"]*" | \
                sed 's,march=,,'))

WARN_CFLAGS = -pedantic -Wall -Wextra
CFLAGS += -std=gnu99 $(WARN_CFLAGS)

ifeq ($(DEBUG), 0)
  OPT_CFLAGS := -O3 -funroll-loops -ffast-math -DNDEBUG
# Options -msse -mfpmath=sse improve performance but are not portable.
# Options -fstandard-precision=fast -ftree-vectorize are not well
# supported in some versions/architectures.
else
  CFLAGS += -g3
endif

MISSING_MARCH_BEGIN=Cannot guess cpu type for gcc
MISSING_MARCH_END=. Please specify your cpu type, \
'make march=i686' typically works fine in most computers. \
For more fine tuning, consult compiler manual.

ifndef MARCH
  ifneq ($(GCC_VERSION),1)
    $(error Please upgrade to a recent version of GCC (>= 4.3))
  else
    MARCH := native
    ARCH := $(gcc-guess-march)
    OPT_CFLAGS += -march=$(ARCH)
  endif
else
  ARCH := $(gcc-guess-march)
  OPT_CFLAGS += -march=$(ARCH)
endif
