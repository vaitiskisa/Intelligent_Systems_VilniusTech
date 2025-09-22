.PHONY: all clean
.DEFAULT_GOAL := all

CC ?= gcc
SRC_DIRS ?= src
INCLUDE_DIRS ?= include
CPPFLAGS += $(addprefix -I,$(INCLUDE_DIRS)) -MMD -MP

rwildcard = $(foreach dir,$(wildcard $(1:=/*)),$(call rwildcard,$(dir),$(2)) $(filter $(subst *,%,$(2)),$(dir)))

ifneq ($(wildcard /usr/lib/clang),)
    ifneq ($(wildcard /usr/bin/clang),)
	CLANG_VERSION := $(shell clang --version | sed -n 's/^clang version \([0-9.]*\).*/\1/p' | cut -d. -f1)
	CLANG_DIR := /usr/lib/clang/$(CLANG_VERSION)/lib/linux

	ifneq ($(wildcard $(CLANG_DIR)),)
		CFLAGS := $(CFLAGS) -fsanitize=leak -fsanitize=address # -fsanitize=undefined
		SANITIZER_BIN := $(CLANG_DIR)/libclang_rt.asan-x86_64.a $(CLANG_DIR)/libclang_rt.ubsan_standalone_cxx-x86_64.a
		CLANG_SANITIZER := 1
	endif
    endif
endif

ifndef CLANG_SANITIZER
    CFLAGS := $(CFLAGS) -fsanitize=leak -fsanitize=address
    CXXFLAGS := $(CXXFLAGS) -fsanitize=leak -fsanitize=address
    LDFLAGS := $(LDFLAGS) -lasan
endif

BUILD_PATH := ./build
BUILD_TARGET := $(BUILD_PATH)/build_run

CFLAGS := $(CFLAGS) -c -g -O0 -Wall -Wno-format-overflow -D_GNU_SOURCE
CFLAGS := $(CFLAGS) -Wno-address-of-packed-member 
LDFLAGS := $(LDFLAGS) -lm -lpthread -lbacktrace -rdynamic

CSRCS := $(shell find $(SRC_DIRS) -name '*.c')

OBJS := $(OBJS) $(patsubst %.c,$(BUILD_PATH)/%.o, $(CSRCS))

all: $(BUILD_TARGET)

$(BUILD_TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(OBJS) -o $@ $(LDFLAGS) $(SANITIZER_BIN)

$(BUILD_PATH)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

-include $(call rwildcard, $(BUILD_PATH), *.d)

run: $(BUILD_TARGET)
	$(BUILD_TARGET)

clean:
	rm -rf $(BUILD_PATH)
