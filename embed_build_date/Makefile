# 
# make PLATFORM=<platform name or path to platform>
# 

TARGET		=  embed_build_date
TARGET_HW	=  ${TARGET}.elf 
TARGET_SW	=  ${TARGET}_sw.elf

APPSOURCES = main.cpp


# Clock IDs
CLOCK_ID    := 2
DM_CLOCK_ID	:= 2

# SDSoC-related
PLATFORM    := zc702
SDSFLAGS	= -sds-sys-config Linux -sds-pf ${PLATFORM} -dmclkid ${DM_CLOCK_ID}
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
	${CC} ${CFLAGS} -sds-hw get_build_date main.cpp -clkid ${CLOCK_ID} -sds-end $< 


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