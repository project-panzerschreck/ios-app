#import "RMQRScannerViewController.h"
#import <AVFoundation/AVFoundation.h>

@interface RMQRScannerViewController () <AVCaptureMetadataOutputObjectsDelegate>

@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;

@end

@implementation RMQRScannerViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor blackColor];
    self.captureSession = [[AVCaptureSession alloc] init];
    [self configureSession];
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    self.previewLayer.frame = self.view.bounds;
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self startIfAuthorized];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    if (self.captureSession.isRunning) {
        [self.captureSession stopRunning];
    }
}

- (void)configureSession {
    AVCaptureDevice *camera = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if (camera == nil) {
        if (self.onFailure != nil) {
            self.onFailure(@"Camera unavailable on this device.");
        }
        return;
    }

    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:camera error:&error];
    if (input == nil || ![self.captureSession canAddInput:input]) {
        if (self.onFailure != nil) {
            self.onFailure(@"Failed to open camera input.");
        }
        return;
    }
    [self.captureSession addInput:input];

    AVCaptureMetadataOutput *output = [[AVCaptureMetadataOutput alloc] init];
    if (![self.captureSession canAddOutput:output]) {
        if (self.onFailure != nil) {
            self.onFailure(@"Failed to start QR scanner.");
        }
        return;
    }
    [self.captureSession addOutput:output];
    [output setMetadataObjectsDelegate:self queue:dispatch_get_main_queue()];
    output.metadataObjectTypes = @[ AVMetadataObjectTypeQRCode ];

    AVCaptureVideoPreviewLayer *previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
    previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    previewLayer.frame = self.view.bounds;
    [self.view.layer addSublayer:previewLayer];
    self.previewLayer = previewLayer;
}

- (void)startIfAuthorized {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    if (status == AVAuthorizationStatusAuthorized) {
        if (!self.captureSession.isRunning) {
            [self.captureSession startRunning];
        }
        return;
    }

    if (status == AVAuthorizationStatusNotDetermined) {
        __weak typeof(self) weakSelf = self;
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                if (granted) {
                    [weakSelf.captureSession startRunning];
                } else if (weakSelf.onFailure != nil) {
                    weakSelf.onFailure(@"Camera permission denied.");
                }
            });
        }];
        return;
    }

    if (self.onFailure != nil) {
        self.onFailure(@"Camera permission denied.");
    }
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputMetadataObjects:(NSArray<__kindof AVMetadataObject *> *)metadataObjects
       fromConnection:(AVCaptureConnection *)connection {
    AVMetadataMachineReadableCodeObject *object = (AVMetadataMachineReadableCodeObject *)metadataObjects.firstObject;
    NSString *code = object.stringValue;
    if (code.length == 0) {
        return;
    }
    if (self.captureSession.isRunning) {
        [self.captureSession stopRunning];
    }
    if (self.onCodeScanned != nil) {
        self.onCodeScanned(code);
    }
}

@end
