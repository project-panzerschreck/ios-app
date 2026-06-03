TARGET := iphone:clang:10.3:10.3
ARCHS := armv7 armv7s
INSTALL_TARGET_PROCESSES = rmclusternode
PACKAGE_VERSION := 1.0.0
ARMV7_REBUILD_DIR := $(CURDIR)/armv7-rebuild-ios10
ARMV7_LIB_DIR := $(ARMV7_REBUILD_DIR)/lib
ARMV7_VENDOR_DIR := $(ARMV7_REBUILD_DIR)/vendor/llama.cpp
ARMV7_MANIFEST := $(ARMV7_REBUILD_DIR)/manifest.txt
IPA_OUTPUT_DIR := $(CURDIR)/packages
IPA_PAYLOAD_DIR := $(CURDIR)/.ipa-payload
MODULE_CACHE_DIR := $(CURDIR)/.theos/module-cache
LEGACY_TOOLCHAIN_BIN ?= $(HOME)/ios-legacy-toolchain/bin
LEGACY_TOOLCHAIN_LIBCXX ?= $(HOME)/ios-legacy-toolchain/include/c++/v1
IOS10_SDK_ROOT ?= $(HOME)/theos/sdks/iPhoneOS10.3.sdk
IOS10_SDK_LIB_DIR := $(IOS10_SDK_ROOT)/usr/lib

THEOS ?= $(HOME)/theos
ifeq ($(wildcard $(THEOS)/makefiles/common.mk),)
$(error Set THEOS to your Theos root; expected $(THEOS)/makefiles/common.mk)
endif

ifeq ($(wildcard $(LEGACY_TOOLCHAIN_BIN)/clang),)
$(warning Legacy toolchain not found at $(LEGACY_TOOLCHAIN_BIN); Theos will fall back to the active Xcode toolchain)
else
PREFIX ?= $(LEGACY_TOOLCHAIN_BIN)/
endif

ifneq ($(wildcard $(IOS10_SDK_ROOT)),$(IOS10_SDK_ROOT))
$(warning iOS 10.3 SDK not found at $(IOS10_SDK_ROOT); set IOS10_SDK_ROOT to your device SDK path)
else
SYSROOT ?= $(IOS10_SDK_ROOT)
ISYSROOT ?= $(IOS10_SDK_ROOT)
endif

include $(THEOS)/makefiles/common.mk

ifneq ($(wildcard $(LEGACY_TOOLCHAIN_BIN)/clang),)
override TARGET_LD = $(CURDIR)/scripts/legacy-link.sh
export TARGET_LD
endif

APPLICATION_NAME = rmclusternode
IPA_NAME := $(APPLICATION_NAME)_$(PACKAGE_VERSION)_unsigned.ipa

rmclusternode_FILES = \
	distributed-ml-ggml-client-ios/main.m \
	distributed-ml-ggml-client-ios/AppDelegate.m \
	distributed-ml-ggml-client-ios/Diagnostics/AppDiagnostics.m \
	distributed-ml-ggml-client-ios/RMAppLogger.m \
	distributed-ml-ggml-client-ios/RMChatMessage.m \
	distributed-ml-ggml-client-ios/RMConnectionBootstrapPayload.m \
	distributed-ml-ggml-client-ios/RMInferenceService.m \
	distributed-ml-ggml-client-ios/RMLogsViewController.m \
	distributed-ml-ggml-client-ios/RMQRScannerViewController.m \
	distributed-ml-ggml-client-ios/RMRootViewController.m \
	distributed-ml-ggml-client-ios/RMRpcSettings.m \
	distributed-ml-ggml-client-ios/RMStorageServer.m \
	distributed-ml-ggml-client-ios/Bridge/LlamaBridge.mm

rmclusternode_FRAMEWORKS = UIKit Foundation AVFoundation Metal Accelerate
rmclusternode_PRIVATE_FRAMEWORKS =
rmclusternode_LIBRARIES = c++ c++abi
rmclusternode_CFLAGS = \
	-fobjc-arc \
	-fno-modules \
	-fmodules-cache-path=$(MODULE_CACHE_DIR) \
	-isystem $(LEGACY_TOOLCHAIN_LIBCXX) \
	-Idistributed-ml-ggml-client-ios \
	-I$(ARMV7_VENDOR_DIR)/include \
	-I$(ARMV7_VENDOR_DIR)/ggml/include
rmclusternode_CCFLAGS = \
	-std=gnu++14 \
	-fno-modules \
	-fmodules-cache-path=$(MODULE_CACHE_DIR) \
	-isystem $(LEGACY_TOOLCHAIN_LIBCXX) \
	-Idistributed-ml-ggml-client-ios \
	-I$(ARMV7_VENDOR_DIR)/include \
	-I$(ARMV7_VENDOR_DIR)/ggml/include
rmclusternode_LDFLAGS += \
	$(IOS10_SDK_LIB_DIR)/libSystem.tbd \
	$(IOS10_SDK_LIB_DIR)/libgcc_s.1.tbd \
	$(ARMV7_LIB_DIR)/libllama.a \
	$(ARMV7_LIB_DIR)/libggml.a \
	$(ARMV7_LIB_DIR)/libggml-base.a \
	$(ARMV7_LIB_DIR)/libggml-cpu.a \
	$(ARMV7_LIB_DIR)/libggml-blas.a \
	$(ARMV7_LIB_DIR)/libggml-rpc.a
rmclusternode_INFOPLIST = distributed-ml-ggml-client-ios/Info.plist
rmclusternode_RESOURCE_FILES = \
	distributed-ml-ggml-client-ios/Info.plist \
	distributed-ml-ggml-client-ios/Default-568h@2x.png \
	rmcluster-node/Resources/AppIcon29x29.png \
	rmcluster-node/Resources/AppIcon29x29@2x.png \
	rmcluster-node/Resources/AppIcon29x29@3x.png \
	rmcluster-node/Resources/AppIcon40x40.png \
	rmcluster-node/Resources/AppIcon40x40@2x.png \
	rmcluster-node/Resources/AppIcon40x40@3x.png \
	rmcluster-node/Resources/AppIcon50x50.png \
	rmcluster-node/Resources/AppIcon50x50@2x.png \
	rmcluster-node/Resources/AppIcon57x57.png \
	rmcluster-node/Resources/AppIcon57x57@2x.png \
	rmcluster-node/Resources/AppIcon57x57@3x.png \
	rmcluster-node/Resources/AppIcon60x60.png \
	rmcluster-node/Resources/AppIcon60x60@2x.png \
	rmcluster-node/Resources/AppIcon60x60@3x.png \
	rmcluster-node/Resources/AppIcon72x72.png \
	rmcluster-node/Resources/AppIcon72x72@2x.png \
	rmcluster-node/Resources/AppIcon76x76.png \
	rmcluster-node/Resources/AppIcon76x76@2x.png

include $(THEOS_MAKE_PATH)/application.mk

.PHONY: armv7-rebuild-ios10 ipa

$(ARMV7_MANIFEST):
	bash scripts/build-ggml-ios10-armv7.sh

armv7-rebuild-ios10: $(ARMV7_MANIFEST)

before-all:: $(ARMV7_MANIFEST)
	@mkdir -p "$(MODULE_CACHE_DIR)"

internal-ipa:: stage
	@test -d "$(THEOS_STAGING_DIR)/Applications/$(APPLICATION_NAME).app" || (echo "Expected staged app at $(THEOS_STAGING_DIR)/Applications/$(APPLICATION_NAME).app" && exit 1)
	@mkdir -p "$(IPA_OUTPUT_DIR)"
	@rm -rf "$(IPA_PAYLOAD_DIR)"
	@mkdir -p "$(IPA_PAYLOAD_DIR)/Payload"
	@cp -R "$(THEOS_STAGING_DIR)/Applications/$(APPLICATION_NAME).app" "$(IPA_PAYLOAD_DIR)/Payload/"
	@cd "$(IPA_PAYLOAD_DIR)" && zip -qry "$(IPA_OUTPUT_DIR)/$(IPA_NAME)" Payload
	@echo "Created $(IPA_OUTPUT_DIR)/$(IPA_NAME)"

ipa: internal-ipa
