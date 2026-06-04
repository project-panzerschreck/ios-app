#import <Foundation/Foundation.h>

@interface RMStorageServer : NSObject

- (instancetype)initWithStorageDirectory:(NSURL *)storageDirectory;
@property (nonatomic, assign, readonly, getter=isRunning) BOOL running;
@property (nonatomic, assign, readonly, getter=isBusy) BOOL busy;
- (BOOL)startOnPort:(NSInteger)port;
- (void)stop;

@end
