#import <Foundation/Foundation.h>

@interface RMConnectionBootstrapPayload : NSObject

@property (nonatomic, copy) NSString *host;
@property (nonatomic, strong) NSNumber *port;
@property (nonatomic, copy) NSString *token;
@property (nonatomic, copy) NSString *device;

+ (instancetype)payloadWithRawValue:(NSString *)rawValue;

@end
