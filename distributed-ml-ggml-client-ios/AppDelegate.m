#import "AppDelegate.h"
#import "RMRootViewController.h"

NSString * const RMOpenURLNotification = @"RMOpenURLNotification";

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    RMRootViewController *rootViewController = [[RMRootViewController alloc] init];
    self.window.rootViewController = rootViewController;
    [self.window makeKeyAndVisible];

    NSURL *launchURL = launchOptions[UIApplicationLaunchOptionsURLKey];
    if (launchURL != nil) {
        [self postOpenURLNotification:launchURL];
    }
    return YES;
}

- (BOOL)application:(UIApplication *)application openURL:(NSURL *)url options:(NSDictionary<UIApplicationOpenURLOptionsKey,id> *)options {
    [self postOpenURLNotification:url];
    return YES;
}

- (BOOL)application:(UIApplication *)application openURL:(NSURL *)url sourceApplication:(NSString *)sourceApplication annotation:(id)annotation {
    [self postOpenURLNotification:url];
    return YES;
}

- (void)postOpenURLNotification:(NSURL *)url {
    if (url == nil) {
        return;
    }
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:RMOpenURLNotification
                                                            object:nil
                                                          userInfo:@{ @"url" : url.absoluteString ?: @"" }];
    });
}

@end
