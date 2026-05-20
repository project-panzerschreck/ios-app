#import "RMChatMessage.h"

@implementation RMChatMessage

- (instancetype)initWithRole:(NSString *)role content:(NSString *)content {
    self = [super init];
    if (self) {
        _role = [role copy] ?: @"";
        _content = [content copy] ?: @"";
    }
    return self;
}

@end
