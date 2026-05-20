#import <UIKit/UIKit.h>

@interface RMQRScannerViewController : UIViewController

@property (nonatomic, copy) void (^onCodeScanned)(NSString *code);
@property (nonatomic, copy) void (^onFailure)(NSString *message);

@end
