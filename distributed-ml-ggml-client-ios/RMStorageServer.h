#import <Foundation/Foundation.h>

@interface RMStorageServer : NSObject

- (instancetype)initWithStorageDirectory:(NSURL *)storageDirectory;
- (BOOL)startOnPort:(NSInteger)port;
- (void)stop;

@end
