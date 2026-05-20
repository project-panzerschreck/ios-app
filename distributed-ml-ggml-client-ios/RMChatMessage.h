#import <Foundation/Foundation.h>

@interface RMChatMessage : NSObject

@property (nonatomic, copy) NSString *role;
@property (nonatomic, copy) NSString *content;

- (instancetype)initWithRole:(NSString *)role content:(NSString *)content;

@end
