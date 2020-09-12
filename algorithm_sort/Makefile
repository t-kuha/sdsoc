# How to build:
#   make
# or 
#   make PLATFORM=<platform name or path to platform> CLOCK_ID=<clock ID> DM_CLOCK_ID=<clock ID> SYS_CFG=<system configuration>
# 

TARGET		=  algo_sort
TARGET_HW	=  ${TARGET}.elf 
TARGET_SW	=  ${TARGET}_sw.elf

APPSOURCES = main.cpp hw.cpp


# Clock IDs
CLOCK_ID    := 2
DM_CLOCK_ID	:= 2

# SDSoC-related
PLATFORM    := zc702
SYS_CFG		:= linux
SDSFLAGS	= -sds-sys-config ${SYS_CFG} -sds-pf ${PLATFORM} -dmclkid ${DM_CLOCK_ID}
CC			:= sds++ ${SDSFLAGS}

# General option
INCDIRS = -I.
LDDIRS =
LDLIBS =
CFLAGS = -Wall -g -c ${INCDIRS}
LDFLAGS = -g ${LDDIRS} ${LDLIBS}
 
SOURCES := $(patsubst %,.%,$(APPSOURCES))
OBJECTS := $(APPSOURCES:.cpp=.o)


# ----------
.PHONY: all
all: ${TARGET_HW}

${TARGET_HW}: ${OBJECTS}
	${CC} ${OBJECTS} -o $@ ${LDFLAGS}
 
%.o: %.cpp
	${CC} ${CFLAGS} -sds-hw ex_sort hw.cpp -clkid ${CLOCK_ID} -sds-end $< 


# No HW acceleration
${TARGET_SW}:
	${CC} ${SDSFLAGS} ${CFLAGS} \
	main.cpp -o $@


.PHONY: clean
clean:
	-$(RM) -rf .Xil
	-$(RM) -rf _sds
	-$(RM) -rf sd_card
	-$(RM) -f *.elf
	-$(RM) -f *.bit
	-$(RM) -f *.o