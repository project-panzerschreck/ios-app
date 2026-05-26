#import "RMRootViewController.h"
#import "AppDelegate.h"
#import "RMChatMessage.h"
#import "RMConnectionBootstrapPayload.h"
#import "RMInferenceService.h"
#import "RMQRScannerViewController.h"
#import "RMRpcSettings.h"
#import <objc/runtime.h>

@interface RMRootViewController () <UIDocumentPickerDelegate, UITextFieldDelegate>

@property (nonatomic, strong) RMInferenceService *service;
@property (nonatomic, strong) RMRpcSettings *settings;
@property (nonatomic, strong) UISegmentedControl *segmentControl;
@property (nonatomic, strong) UIScrollView *scrollView;
@property (nonatomic, strong) UIStackView *contentStack;
@property (nonatomic, strong) UIStackView *inferencePane;
@property (nonatomic, strong) UIStackView *rpcPane;

@property (nonatomic, strong) UIStackView *modelButtonsStack;
@property (nonatomic, strong) UILabel *modelStatusLabel;
@property (nonatomic, strong) UIButton *loadOtherButton;
@property (nonatomic, strong) UIButton *unloadButton;
@property (nonatomic, strong) UITextView *conversationTextView;
@property (nonatomic, strong) UILabel *tokensPerSecondLabel;
@property (nonatomic, strong) UITextView *chatInputView;
@property (nonatomic, strong) UIButton *sendButton;
@property (nonatomic, strong) UIButton *clearConversationButton;
@property (nonatomic, strong) UILabel *maxTokensValueLabel;
@property (nonatomic, strong) UIStepper *maxTokensStepper;
@property (nonatomic, strong) UILabel *temperatureValueLabel;
@property (nonatomic, strong) UISlider *temperatureSlider;

@property (nonatomic, strong) UITextView *endpointsTextView;
@property (nonatomic, strong) UITextField *connectionStringField;
@property (nonatomic, strong) UITextField *serverURLField;
@property (nonatomic, strong) UITextField *serverHostField;
@property (nonatomic, strong) UITextField *tokenField;
@property (nonatomic, strong) UITextField *hostField;
@property (nonatomic, strong) UITextField *serverPortField;
@property (nonatomic, strong) UITextField *threadCountField;
@property (nonatomic, strong) UITextField *portField;
@property (nonatomic, strong) UITextField *storagePortField;
@property (nonatomic, strong) UIStepper *serverPortStepper;
@property (nonatomic, strong) UIStepper *threadCountStepper;
@property (nonatomic, strong) UIStepper *portStepper;
@property (nonatomic, strong) UIStepper *storagePortStepper;
@property (nonatomic, strong) UILabel *importStatusLabel;
@property (nonatomic, strong) UIButton *rpcStartStopButton;
@property (nonatomic, strong) UILabel *rpcStatusLabel;

@property (nonatomic, assign) NSInteger contextLength;

@end

@implementation RMRootViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.service = [RMInferenceService sharedService];
    self.settings = [RMRpcSettings sharedSettings];
    self.contextLength = 1024;
    self.view.backgroundColor = [UIColor whiteColor];
    [self buildUI];
    [self loadSettingsIntoFields];
    [self refreshLocalModels];
    [self refreshUI];

    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(inferenceServiceDidUpdate:) name:RMInferenceServiceDidUpdateNotification object:self.service];
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(handleOpenURLNotification:) name:RMOpenURLNotification object:nil];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)buildUI {
    self.segmentControl = [[UISegmentedControl alloc] initWithItems:@[ @"Inference", @"GGML RPC Worker" ]];
    self.segmentControl.selectedSegmentIndex = 0;
    [self.segmentControl addTarget:self action:@selector(segmentChanged:) forControlEvents:UIControlEventValueChanged];
    self.segmentControl.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.segmentControl];

    self.scrollView = [[UIScrollView alloc] init];
    self.scrollView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.scrollView];

    self.contentStack = [[UIStackView alloc] init];
    self.contentStack.axis = UILayoutConstraintAxisVertical;
    self.contentStack.spacing = 16.0;
    self.contentStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.scrollView addSubview:self.contentStack];

    self.inferencePane = [self verticalStack];
    self.rpcPane = [self verticalStack];
    [self.contentStack addArrangedSubview:self.inferencePane];
    [self.contentStack addArrangedSubview:self.rpcPane];
    self.rpcPane.hidden = YES;

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    NSLayoutYAxisAnchor *topAnchor = self.topLayoutGuide.bottomAnchor;
    NSLayoutYAxisAnchor *bottomAnchor = self.bottomLayoutGuide.topAnchor;
    #pragma clang diagnostic pop
    if (@available(iOS 11.0, *)) {
        topAnchor = self.view.safeAreaLayoutGuide.topAnchor;
        bottomAnchor = self.view.safeAreaLayoutGuide.bottomAnchor;
    }

    [NSLayoutConstraint activateConstraints:@[
        [self.segmentControl.topAnchor constraintEqualToAnchor:topAnchor constant:12.0],
        [self.segmentControl.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16.0],
        [self.segmentControl.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-16.0],

        [self.scrollView.topAnchor constraintEqualToAnchor:self.segmentControl.bottomAnchor constant:12.0],
        [self.scrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.scrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.scrollView.bottomAnchor constraintEqualToAnchor:bottomAnchor],

        [self.contentStack.topAnchor constraintEqualToAnchor:self.scrollView.topAnchor constant:16.0],
        [self.contentStack.leadingAnchor constraintEqualToAnchor:self.scrollView.leadingAnchor constant:16.0],
        [self.contentStack.trailingAnchor constraintEqualToAnchor:self.scrollView.trailingAnchor constant:-16.0],
        [self.contentStack.bottomAnchor constraintEqualToAnchor:self.scrollView.bottomAnchor constant:-24.0],
        [self.contentStack.widthAnchor constraintEqualToAnchor:self.view.widthAnchor constant:-32.0],
    ]];

    [self buildInferencePane];
    [self buildRPCPane];
}

- (UIStackView *)verticalStack {
    UIStackView *stack = [[UIStackView alloc] init];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 10.0;
    stack.alignment = UIStackViewAlignmentFill;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    return stack;
}

- (UIView *)sectionContainerWithTitle:(NSString *)title {
    UIStackView *stack = [self verticalStack];
    stack.spacing = 8.0;
    UILabel *label = [[UILabel alloc] init];
    label.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    label.text = title;
    [stack addArrangedSubview:label];
    UIView *container = [[UIView alloc] init];
    container.backgroundColor = [UIColor colorWithWhite:0.95 alpha:1.0];
    container.layer.cornerRadius = 8.0;
    container.translatesAutoresizingMaskIntoConstraints = NO;
    [stack addArrangedSubview:container];
    return stack;
}

- (void)buildInferencePane {
    UILabel *modelSectionTitle = [[UILabel alloc] init];
    modelSectionTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    modelSectionTitle.text = @"Model";
    [self.inferencePane addArrangedSubview:modelSectionTitle];

    self.modelButtonsStack = [self verticalStack];
    [self.inferencePane addArrangedSubview:self.modelButtonsStack];

    self.loadOtherButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.loadOtherButton setTitle:@"Load other…" forState:UIControlStateNormal];
    [self.loadOtherButton addTarget:self action:@selector(loadOtherTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.inferencePane addArrangedSubview:self.loadOtherButton];

    self.modelStatusLabel = [[UILabel alloc] init];
    self.modelStatusLabel.numberOfLines = 0;
    self.modelStatusLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    [self.inferencePane addArrangedSubview:self.modelStatusLabel];

    self.unloadButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.unloadButton setTitle:@"Unload model" forState:UIControlStateNormal];
    [self.unloadButton addTarget:self action:@selector(unloadModelTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.inferencePane addArrangedSubview:self.unloadButton];

    UILabel *conversationTitle = [[UILabel alloc] init];
    conversationTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    conversationTitle.text = @"Conversation";
    [self.inferencePane addArrangedSubview:conversationTitle];

    self.conversationTextView = [[UITextView alloc] init];
    self.conversationTextView.editable = NO;
    self.conversationTextView.scrollEnabled = NO;
    self.conversationTextView.layer.cornerRadius = 8.0;
    self.conversationTextView.backgroundColor = [UIColor colorWithWhite:0.95 alpha:1.0];
    self.conversationTextView.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    self.conversationTextView.textContainerInset = UIEdgeInsetsMake(12, 10, 12, 10);
    self.conversationTextView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.conversationTextView.heightAnchor constraintGreaterThanOrEqualToConstant:160.0].active = YES;
    [self.inferencePane addArrangedSubview:self.conversationTextView];

    self.tokensPerSecondLabel = [[UILabel alloc] init];
    self.tokensPerSecondLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    self.tokensPerSecondLabel.textAlignment = NSTextAlignmentRight;
    [self.inferencePane addArrangedSubview:self.tokensPerSecondLabel];

    UILabel *messageLabel = [[UILabel alloc] init];
    messageLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    messageLabel.text = @"Message…";
    [self.inferencePane addArrangedSubview:messageLabel];

    self.chatInputView = [[UITextView alloc] init];
    self.chatInputView.layer.cornerRadius = 8.0;
    self.chatInputView.backgroundColor = [UIColor colorWithWhite:0.95 alpha:1.0];
    self.chatInputView.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    self.chatInputView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.chatInputView.heightAnchor constraintEqualToConstant:88.0].active = YES;
    [self.inferencePane addArrangedSubview:self.chatInputView];

    UIStackView *messageActions = [[UIStackView alloc] init];
    messageActions.axis = UILayoutConstraintAxisHorizontal;
    messageActions.spacing = 10.0;
    self.sendButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.sendButton setTitle:@"Send" forState:UIControlStateNormal];
    [self.sendButton addTarget:self action:@selector(sendTapped:) forControlEvents:UIControlEventTouchUpInside];
    self.clearConversationButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.clearConversationButton setTitle:@"Clear conversation" forState:UIControlStateNormal];
    [self.clearConversationButton addTarget:self action:@selector(clearConversationTapped:) forControlEvents:UIControlEventTouchUpInside];
    [messageActions addArrangedSubview:self.sendButton];
    [messageActions addArrangedSubview:self.clearConversationButton];
    [self.inferencePane addArrangedSubview:messageActions];

    UILabel *parameterTitle = [[UILabel alloc] init];
    parameterTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    parameterTitle.text = @"Parameters";
    [self.inferencePane addArrangedSubview:parameterTitle];

    UIStackView *maxTokensRow = [[UIStackView alloc] init];
    maxTokensRow.axis = UILayoutConstraintAxisHorizontal;
    maxTokensRow.spacing = 10.0;
    UILabel *maxTokensTitle = [[UILabel alloc] init];
    maxTokensTitle.text = @"Max tokens";
    self.maxTokensValueLabel = [[UILabel alloc] init];
    self.maxTokensStepper = [[UIStepper alloc] init];
    self.maxTokensStepper.minimumValue = 1;
    self.maxTokensStepper.maximumValue = 2048;
    self.maxTokensStepper.stepValue = 50;
    self.maxTokensStepper.value = 200;
    [self.maxTokensStepper addTarget:self action:@selector(maxTokensChanged:) forControlEvents:UIControlEventValueChanged];
    [maxTokensRow addArrangedSubview:maxTokensTitle];
    [maxTokensRow addArrangedSubview:self.maxTokensValueLabel];
    [maxTokensRow addArrangedSubview:self.maxTokensStepper];
    [self.inferencePane addArrangedSubview:maxTokensRow];

    UILabel *temperatureTitle = [[UILabel alloc] init];
    temperatureTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    [self.inferencePane addArrangedSubview:temperatureTitle];
    self.temperatureValueLabel = temperatureTitle;
    self.temperatureSlider = [[UISlider alloc] init];
    self.temperatureSlider.minimumValue = 0.0f;
    self.temperatureSlider.maximumValue = 2.0f;
    self.temperatureSlider.value = 0.8f;
    [self.temperatureSlider addTarget:self action:@selector(temperatureChanged:) forControlEvents:UIControlEventValueChanged];
    [self.inferencePane addArrangedSubview:self.temperatureSlider];
}

- (void)buildRPCPane {
    UILabel *endpointTitle = [[UILabel alloc] init];
    endpointTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    endpointTitle.text = @"Endpoints";
    [self.rpcPane addArrangedSubview:endpointTitle];

    self.endpointsTextView = [[UITextView alloc] init];
    self.endpointsTextView.editable = NO;
    self.endpointsTextView.scrollEnabled = NO;
    self.endpointsTextView.layer.cornerRadius = 8.0;
    self.endpointsTextView.backgroundColor = [UIColor colorWithWhite:0.95 alpha:1.0];
    self.endpointsTextView.font = [UIFont fontWithName:@"Menlo-Regular" size:13.0] ?: [UIFont systemFontOfSize:13.0];
    self.endpointsTextView.textContainerInset = UIEdgeInsetsMake(12, 10, 12, 10);
    [self.endpointsTextView.heightAnchor constraintGreaterThanOrEqualToConstant:120.0].active = YES;
    [self.rpcPane addArrangedSubview:self.endpointsTextView];

    UILabel *connectionTitle = [[UILabel alloc] init];
    connectionTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    connectionTitle.text = @"Coordinator";
    [self.rpcPane addArrangedSubview:connectionTitle];

    self.connectionStringField = [self textFieldWithPlaceholder:@"Paste connection string or rmcluster:// URL"];
    self.serverURLField = [self textFieldWithPlaceholder:@"Coordinator URL"];
    self.serverHostField = [self textFieldWithPlaceholder:@"Coordinator host"];
    self.tokenField = [self textFieldWithPlaceholder:@"Token"];
    self.hostField = [self textFieldWithPlaceholder:@"0.0.0.0"];
    self.serverPortField = [self numericField];
    self.threadCountField = [self numericField];
    self.portField = [self numericField];
    self.storagePortField = [self numericField];

    [self.rpcPane addArrangedSubview:self.connectionStringField];

    UIButton *applyConnectionButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [applyConnectionButton setTitle:@"Apply connection string" forState:UIControlStateNormal];
    [applyConnectionButton addTarget:self action:@selector(applyConnectionStringTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.rpcPane addArrangedSubview:applyConnectionButton];

    [self.rpcPane addArrangedSubview:self.serverURLField];
    [self.rpcPane addArrangedSubview:self.serverHostField];
    self.serverPortStepper = [[UIStepper alloc] init];
    [self.serverPortStepper addTarget:self action:@selector(serverPortStepperChanged:) forControlEvents:UIControlEventValueChanged];
    [self.rpcPane addArrangedSubview:[self numericRowWithTitle:@"Coordinator port" field:self.serverPortField stepper:self.serverPortStepper]];
    [self.rpcPane addArrangedSubview:self.tokenField];

    UIButton *scanQRButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [scanQRButton setTitle:@"Scan QR code" forState:UIControlStateNormal];
    [scanQRButton addTarget:self action:@selector(scanQRTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.rpcPane addArrangedSubview:scanQRButton];

    UIButton *clipboardButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [clipboardButton setTitle:@"Import from clipboard" forState:UIControlStateNormal];
    [clipboardButton addTarget:self action:@selector(importFromClipboardTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.rpcPane addArrangedSubview:clipboardButton];

    self.importStatusLabel = [[UILabel alloc] init];
    self.importStatusLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    self.importStatusLabel.numberOfLines = 0;
    [self.rpcPane addArrangedSubview:self.importStatusLabel];

    self.threadCountStepper = [[UIStepper alloc] init];
    [self.threadCountStepper addTarget:self action:@selector(threadStepperChanged:) forControlEvents:UIControlEventValueChanged];
    [self.rpcPane addArrangedSubview:[self numericRowWithTitle:@"Thread count" field:self.threadCountField stepper:self.threadCountStepper]];
    [self.rpcPane addArrangedSubview:[self labeledFieldRowWithTitle:@"Host" field:self.hostField]];
    self.portStepper = [[UIStepper alloc] init];
    [self.portStepper addTarget:self action:@selector(portStepperChanged:) forControlEvents:UIControlEventValueChanged];
    [self.rpcPane addArrangedSubview:[self numericRowWithTitle:@"Port" field:self.portField stepper:self.portStepper]];
    self.storagePortStepper = [[UIStepper alloc] init];
    [self.storagePortStepper addTarget:self action:@selector(storagePortStepperChanged:) forControlEvents:UIControlEventValueChanged];
    [self.rpcPane addArrangedSubview:[self numericRowWithTitle:@"Storage Port" field:self.storagePortField stepper:self.storagePortStepper]];

    UILabel *footer = [[UILabel alloc] init];
    footer.numberOfLines = 0;
    footer.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    footer.text = @"Paste a rmcluster://connect URL, scan a QR code, or manually edit the coordinator host, port, and token.";
    [self.rpcPane addArrangedSubview:footer];

    self.rpcStatusLabel = [[UILabel alloc] init];
    self.rpcStatusLabel.numberOfLines = 0;
    self.rpcStatusLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    [self.rpcPane addArrangedSubview:self.rpcStatusLabel];

    self.rpcStartStopButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.rpcStartStopButton addTarget:self action:@selector(rpcStartStopTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.rpcPane addArrangedSubview:self.rpcStartStopButton];
}

- (UITextField *)textFieldWithPlaceholder:(NSString *)placeholder {
    UITextField *field = [[UITextField alloc] init];
    field.borderStyle = UITextBorderStyleRoundedRect;
    field.placeholder = placeholder;
    field.delegate = self;
    return field;
}

- (UITextField *)numericField {
    UITextField *field = [self textFieldWithPlaceholder:nil];
    field.keyboardType = UIKeyboardTypeNumbersAndPunctuation;
    field.textAlignment = NSTextAlignmentRight;
    return field;
}

- (UIView *)labeledFieldRowWithTitle:(NSString *)title field:(UITextField *)field {
    UIStackView *row = [[UIStackView alloc] init];
    row.axis = UILayoutConstraintAxisHorizontal;
    row.spacing = 10.0;
    row.alignment = UIStackViewAlignmentCenter;
    UILabel *label = [[UILabel alloc] init];
    label.text = title;
    [field.widthAnchor constraintEqualToConstant:180.0].active = YES;
    [row addArrangedSubview:label];
    [row addArrangedSubview:field];
    return row;
}

- (UIView *)numericRowWithTitle:(NSString *)title field:(UITextField *)field stepper:(UIStepper *)stepper {
    UIStackView *row = [[UIStackView alloc] init];
    row.axis = UILayoutConstraintAxisHorizontal;
    row.spacing = 10.0;
    row.alignment = UIStackViewAlignmentCenter;
    UILabel *label = [[UILabel alloc] init];
    label.text = title;
    stepper.minimumValue = 1;
    stepper.maximumValue = 65535;
    [field.widthAnchor constraintEqualToConstant:80.0].active = YES;
    [row addArrangedSubview:label];
    [row addArrangedSubview:field];
    [row addArrangedSubview:stepper];
    return row;
}

- (void)loadSettingsIntoFields {
    NSString *coordinatorHost = self.settings.clusterServerHost.length > 0 ? self.settings.clusterServerHost : self.settings.discoveryIp;
    NSInteger coordinatorPort = self.settings.clusterServerPort > 0 ? self.settings.clusterServerPort : self.settings.discoveryPort;
    if (coordinatorPort == 0) {
        coordinatorPort = 4917;
    }
    self.serverHostField.text = coordinatorHost ?: @"";
    self.serverPortField.text = [NSString stringWithFormat:@"%ld", (long)coordinatorPort];
    self.serverPortStepper.value = coordinatorPort;
    self.tokenField.text = self.settings.clusterToken;
    self.hostField.text = self.settings.host;
    self.portField.text = [NSString stringWithFormat:@"%ld", (long)self.settings.port];
    self.portStepper.value = self.settings.port;
    self.storagePortField.text = [NSString stringWithFormat:@"%ld", (long)self.settings.storagePort];
    self.storagePortStepper.value = self.settings.storagePort;
    self.threadCountField.text = [NSString stringWithFormat:@"%ld", (long)self.settings.threads];
    self.threadCountStepper.minimumValue = 1;
    self.threadCountStepper.maximumValue = 64;
    self.threadCountStepper.value = self.settings.threads;
    self.maxTokensValueLabel.text = @"200";
    [self updateTemperatureLabel];
    [self syncDiscoverySettingsFromCoordinatorFields];
    [self syncServerURLFromHostAndPort];
}

- (void)refreshLocalModels {
    for (UIView *view in self.modelButtonsStack.arrangedSubviews) {
        [self.modelButtonsStack removeArrangedSubview:view];
        [view removeFromSuperview];
    }

    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtURL:documentsDirectory includingPropertiesForKeys:nil options:0 error:nil];
    NSMutableArray<NSURL *> *models = [NSMutableArray array];
    for (NSURL *fileURL in files) {
        if ([[fileURL.pathExtension lowercaseString] isEqualToString:@"gguf"]) {
            [models addObject:fileURL];
        }
    }
    [models sortUsingComparator:^NSComparisonResult(NSURL *left, NSURL *right) {
        return [left.lastPathComponent compare:right.lastPathComponent];
    }];

    if (models.count == 0) {
        UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
        [button setTitle:@"Load .gguf model…" forState:UIControlStateNormal];
        [button addTarget:self action:@selector(loadOtherTapped:) forControlEvents:UIControlEventTouchUpInside];
        [self.modelButtonsStack addArrangedSubview:button];
        return;
    }

    for (NSURL *fileURL in models) {
        UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
        [button setTitle:fileURL.lastPathComponent forState:UIControlStateNormal];
        button.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeft;
        button.tag = self.modelButtonsStack.arrangedSubviews.count;
        objc_setAssociatedObject(button, @"modelURL", fileURL, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [button addTarget:self action:@selector(localModelTapped:) forControlEvents:UIControlEventTouchUpInside];
        [self.modelButtonsStack addArrangedSubview:button];
    }
}

- (void)refreshUI {
    switch (self.service.modelState) {
        case RMModelStateUnloaded:
            self.modelStatusLabel.text = @"";
            break;
        case RMModelStateLoading:
            self.modelStatusLabel.text = @"Loading model…";
            break;
        case RMModelStateReady: {
            if (self.service.modelInfo != nil) {
                double megabytes = (double)self.service.modelInfo.fileSizeBytes / 1048576.0;
                NSString *sizeText = megabytes >= 1000.0 ? [NSString stringWithFormat:@"%.1f GB", megabytes / 1024.0] : [NSString stringWithFormat:@"%.0f MB", megabytes];
                self.modelStatusLabel.text = [NSString stringWithFormat:@"%@\n%ld layers • embd %ld • ctx %ld • %@",
                                              self.service.modelName ?: @"",
                                              (long)self.service.modelLayers,
                                              (long)self.service.modelInfo.nEmbd,
                                              (long)self.service.modelInfo.nCtx,
                                              sizeText];
            } else {
                self.modelStatusLabel.text = self.service.modelName ?: @"";
            }
            break;
        }
        case RMModelStateGenerating:
            self.modelStatusLabel.text = @"Generating…";
            break;
        case RMModelStateError:
            self.modelStatusLabel.text = self.service.modelErrorMessage ?: @"";
            break;
    }

    self.unloadButton.hidden = !(self.service.modelState == RMModelStateReady || self.service.modelState == RMModelStateGenerating);
    [self.sendButton setTitle:(self.service.modelState == RMModelStateGenerating ? @"Stop" : @"Send") forState:UIControlStateNormal];
    self.chatInputView.editable = (self.service.modelState != RMModelStateGenerating);

    NSMutableString *conversationText = [NSMutableString string];
    for (RMChatMessage *message in self.service.chatMessages) {
        NSString *speaker = [message.role isEqualToString:@"user"] ? @"You" : @"Assistant";
        [conversationText appendFormat:@"%@\n%@\n\n", speaker, message.content.length > 0 ? message.content : @"…"];
    }
    self.conversationTextView.text = conversationText;
    self.conversationTextView.hidden = self.service.chatMessages.count == 0;
    self.clearConversationButton.hidden = self.service.chatMessages.count == 0;

    if (self.service.tokensPerSecond > 0.0) {
        self.tokensPerSecondLabel.text = [NSString stringWithFormat:@"%.1f tok/s", self.service.tokensPerSecond];
    } else {
        self.tokensPerSecondLabel.text = @"";
    }

    [self updateEndpointsText];
    self.rpcStatusLabel.text = self.service.rpcStatusMessage ?: @"";
    switch (self.service.rpcServerState) {
        case RMRPCServerStateStarting:
            [self.rpcStartStopButton setTitle:@"Starting…" forState:UIControlStateNormal];
            self.rpcStartStopButton.enabled = NO;
            break;
        case RMRPCServerStateRunning:
            [self.rpcStartStopButton setTitle:@"Stop RPC server" forState:UIControlStateNormal];
            self.rpcStartStopButton.enabled = YES;
            break;
        case RMRPCServerStateUnavailable:
            [self.rpcStartStopButton setTitle:@"Start RPC server" forState:UIControlStateNormal];
            self.rpcStartStopButton.enabled = YES;
            break;
        default:
            [self.rpcStartStopButton setTitle:@"Start RPC server" forState:UIControlStateNormal];
            self.rpcStartStopButton.enabled = YES;
            break;
    }
}

- (void)updateEndpointsText {
    NSArray<RMLocalInterface *> *interfaces = [RMInferenceService allLocalIPv4Interfaces];
    if (interfaces.count == 0) {
        self.endpointsTextView.text = @"No network interfaces found";
        return;
    }

    NSMutableArray<NSString *> *lines = [NSMutableArray array];
    for (RMLocalInterface *interface in interfaces) {
        [lines addObject:[NSString stringWithFormat:@"RPC %@:%ld", interface.ip, (long)self.settings.port]];
        [lines addObject:[NSString stringWithFormat:@"Storage %@:%ld", interface.ip, (long)self.settings.storagePort]];
        [lines addObject:interface.label ?: @""];
        [lines addObject:@""];
    }
    self.endpointsTextView.text = [lines componentsJoinedByString:@"\n"];
}

- (void)updateTemperatureLabel {
    self.temperatureValueLabel.text = [NSString stringWithFormat:@"Temperature  %.2f", self.temperatureSlider.value];
}

- (void)segmentChanged:(UISegmentedControl *)sender {
    BOOL showingInference = sender.selectedSegmentIndex == 0;
    self.inferencePane.hidden = !showingInference;
    self.rpcPane.hidden = showingInference;
}

- (void)localModelTapped:(UIButton *)sender {
    NSURL *url = objc_getAssociatedObject(sender, @"modelURL");
    [self.service loadModelFromURL:url contextLength:self.contextLength];
}

- (void)loadOtherTapped:(id)sender {
    UIDocumentPickerViewController *picker = [[UIDocumentPickerViewController alloc] initWithDocumentTypes:@[ @"public.data" ]
                                                                                                     inMode:UIDocumentPickerModeImport];
    picker.delegate = self;
    picker.modalPresentationStyle = UIModalPresentationFormSheet;
    [self presentViewController:picker animated:YES completion:nil];
}

- (void)unloadModelTapped:(id)sender {
    [self.service unloadModel];
}

- (void)sendTapped:(id)sender {
    if (self.service.modelState == RMModelStateGenerating) {
        [self.service cancelGeneration];
        return;
    }
    [self.service sendMessage:self.chatInputView.text
                    maxTokens:(NSInteger)self.maxTokensStepper.value
                  temperature:self.temperatureSlider.value];
    self.chatInputView.text = @"";
}

- (void)clearConversationTapped:(id)sender {
    [self.service clearChat];
}

- (void)maxTokensChanged:(UIStepper *)sender {
    self.maxTokensValueLabel.text = [NSString stringWithFormat:@"%ld", (long)sender.value];
}

- (void)temperatureChanged:(UISlider *)sender {
    [self updateTemperatureLabel];
}

- (void)applyConnectionStringTapped:(id)sender {
    [self applyConnectionConfigFromString:self.connectionStringField.text];
}

- (void)scanQRTapped:(id)sender {
    RMQRScannerViewController *scanner = [[RMQRScannerViewController alloc] init];
    __weak typeof(self) weakSelf = self;
    scanner.modalPresentationStyle = UIModalPresentationFullScreen;
    scanner.onCodeScanned = ^(NSString *code) {
        [weakSelf dismissViewControllerAnimated:YES completion:nil];
        [weakSelf applyConnectionConfigFromString:code];
    };
    scanner.onFailure = ^(NSString *message) {
        [weakSelf dismissViewControllerAnimated:YES completion:nil];
        weakSelf.importStatusLabel.text = message;
    };
    [self presentViewController:scanner animated:YES completion:nil];
}

- (void)importFromClipboardTapped:(id)sender {
    NSString *clipboard = [UIPasteboard generalPasteboard].string;
    if (clipboard.length == 0) {
        self.importStatusLabel.text = @"Clipboard does not contain a connection URL.";
        return;
    }
    [self applyConnectionConfigFromString:clipboard];
}

- (void)rpcStartStopTapped:(id)sender {
    [self syncSettingsFromFields];
    if (self.service.rpcServerState == RMRPCServerStateRunning || self.service.rpcServerState == RMRPCServerStateStarting) {
        [self.service stopRPCServer];
    } else {
        [self.service startRPCServerWithHost:self.settings.host
                                        port:self.settings.port
                                 storagePort:self.settings.storagePort
                                 discoveryIp:self.settings.discoveryIp
                               discoveryPort:self.settings.discoveryPort
                              discoveryToken:self.settings.clusterToken
                                     threads:self.settings.threads
                                    deviceId:self.settings.deviceId];
    }
}

- (void)syncSettingsFromFields {
    self.settings.clusterServerHost = self.serverHostField.text ?: @"";
    self.settings.clusterServerPort = [self.serverPortField.text integerValue];
    self.settings.clusterToken = self.tokenField.text ?: @"";
    [self syncDiscoverySettingsFromCoordinatorFields];
    self.settings.host = self.hostField.text ?: @"";
    self.settings.port = [self.portField.text integerValue];
    self.settings.storagePort = [self.storagePortField.text integerValue];
    self.settings.threads = MAX(1, [self.threadCountField.text integerValue]);
}

- (void)applyConnectionConfigFromString:(NSString *)rawValue {
    RMConnectionBootstrapPayload *payload = [RMConnectionBootstrapPayload payloadWithRawValue:rawValue];
    if (payload == nil) {
        self.importStatusLabel.text = @"Could not parse connection data.";
        return;
    }

    self.connectionStringField.text = [rawValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    self.settings.clusterServerHost = payload.host ?: @"";
    self.serverHostField.text = self.settings.clusterServerHost;

    if (payload.port != nil) {
        self.settings.clusterServerPort = payload.port.integerValue;
        self.serverPortField.text = [NSString stringWithFormat:@"%ld", (long)self.settings.clusterServerPort];
        self.serverPortStepper.value = self.settings.clusterServerPort;
    }

    self.settings.clusterToken = payload.token ?: @"";
    self.tokenField.text = self.settings.clusterToken;
    if (payload.device.length > 0) {
        self.settings.clusterDeviceLabel = payload.device;
    }

    [self syncDiscoverySettingsFromCoordinatorFields];
    [self syncServerURLFromHostAndPort];
    self.segmentControl.selectedSegmentIndex = 1;
    [self segmentChanged:self.segmentControl];
    self.importStatusLabel.text = @"";
}

- (void)syncServerURLFromHostAndPort {
    NSString *trimmedHost = [self.serverHostField.text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (trimmedHost.length == 0) {
        self.serverURLField.text = @"";
        return;
    }
    self.serverURLField.text = [NSString stringWithFormat:@"http://%@:%@", trimmedHost, self.serverPortField.text ?: @"0"];
}

- (void)syncConnectionFieldsFromServerURL {
    RMConnectionBootstrapPayload *payload = [RMConnectionBootstrapPayload payloadWithRawValue:self.serverURLField.text];
    if (payload == nil) {
        return;
    }
    self.serverHostField.text = payload.host ?: @"";
    if (payload.port != nil) {
        self.serverPortField.text = [payload.port stringValue];
        self.serverPortStepper.value = payload.port.doubleValue;
    }
    self.tokenField.text = payload.token ?: @"";
    [self syncDiscoverySettingsFromCoordinatorFields];
}

- (void)serverPortStepperChanged:(UIStepper *)sender {
    self.serverPortField.text = [NSString stringWithFormat:@"%ld", (long)sender.value];
    [self syncServerURLFromHostAndPort];
}

- (void)threadStepperChanged:(UIStepper *)sender {
    self.threadCountField.text = [NSString stringWithFormat:@"%ld", (long)sender.value];
}

- (void)portStepperChanged:(UIStepper *)sender {
    self.portField.text = [NSString stringWithFormat:@"%ld", (long)sender.value];
}

- (void)storagePortStepperChanged:(UIStepper *)sender {
    self.storagePortField.text = [NSString stringWithFormat:@"%ld", (long)sender.value];
}

- (void)syncDiscoverySettingsFromCoordinatorFields {
    self.settings.discoveryIp = self.serverHostField.text ?: @"";
    self.settings.discoveryPort = [self.serverPortField.text integerValue];
}

- (void)inferenceServiceDidUpdate:(NSNotification *)notification {
    [self refreshUI];
}

- (void)handleOpenURLNotification:(NSNotification *)notification {
    NSString *urlString = notification.userInfo[@"url"];
    if (urlString.length > 0) {
        [self applyConnectionConfigFromString:urlString];
    }
}

- (BOOL)textFieldShouldReturn:(UITextField *)textField {
    [textField resignFirstResponder];
    if (textField == self.connectionStringField) {
        [self applyConnectionConfigFromString:textField.text];
    } else if (textField == self.serverURLField) {
        [self syncConnectionFieldsFromServerURL];
    } else if (textField == self.serverHostField || textField == self.serverPortField) {
        [self syncServerURLFromHostAndPort];
    }
    return YES;
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
    NSURL *url = urls.firstObject;
    if (url == nil) {
        return;
    }
    BOOL scoped = [url startAccessingSecurityScopedResource];
    NSURL *destination = [[[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject] URLByAppendingPathComponent:url.lastPathComponent];
    [[NSFileManager defaultManager] removeItemAtURL:destination error:nil];
    [[NSFileManager defaultManager] copyItemAtURL:url toURL:destination error:nil];
    if (scoped) {
        [url stopAccessingSecurityScopedResource];
    }
    [self refreshLocalModels];
    [self.service loadModelFromURL:destination contextLength:self.contextLength];
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller didPickDocumentAtURL:(NSURL *)url {
    if (url == nil) {
        return;
    }
    [self documentPicker:controller didPickDocumentsAtURLs:@[ url ]];
}

@end
