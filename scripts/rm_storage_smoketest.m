#import <Foundation/Foundation.h>
#import "../distributed-ml-ggml-client-ios/RMStorageServer.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <storage-dir> <port>\n", argv[0]);
            return 2;
        }

        NSString *directoryPath = [NSString stringWithUTF8String:argv[1]];
        NSInteger port = [[NSString stringWithUTF8String:argv[2]] integerValue];

        RMStorageServer *server = [[RMStorageServer alloc] initWithStorageDirectory:[NSURL fileURLWithPath:directoryPath isDirectory:YES]];
        if (![server startOnPort:port]) {
            fprintf(stderr, "failed to start server on port %ld\n", (long)port);
            return 1;
        }

        fprintf(stdout, "RMStorageServer listening on 127.0.0.1:%ld\n", (long)port);
        fflush(stdout);
        [[NSRunLoop currentRunLoop] run];
    }
    return 0;
}
