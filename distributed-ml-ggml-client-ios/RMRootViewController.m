#import "RMRootViewController.h"
#import "AppDelegate.h"
#import "RMChatMessage.h"
#import "RMConnectionBootstrapPayload.h"
#import "RMInferenceService.h"
#import "RMQRScannerViewController.h"
#import "RMRpcSettings.h"
#import "RMLogsViewController.h"
#import <objc/runtime.h>

@interface RMRootViewController () <UIDocumentPickerDelegate, UITextFieldDelegate>

@property (nonatomic, strong) RMInferenceService *service;
@property (nonatomic, strong) RMRpcSettings *settings;
@property (nonatomic, strong) UISegmentedControl *segmentControl;
@property (nonatomic, strong) UIScrollView *scrollView;
@property (nonatomic, strong) UIStackView *contentStack;
@property (nonatomic, strong) UIStackView *inferencePane;
@property (nonatomic, strong) UIStackView *rpcPane;
@property (nonatomic, strong) RMLogsViewController *logsViewController;
@property (nonatomic, strong) UIView *logsContainer;

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

@property (nonatomic, assign) BOOL endpointsExpanded;
@property (nonatomic, strong) UIStackView *endpointsStack;
@property (nonatomic, strong) UIButton *endpointsHeaderButton;
@property (nonatomic, strong) UILabel *endpointsChevronLabel;
@property (nonatomic, strong) UITextField *connectionStringField;
@property (nonatomic, strong) UITextField *nicknameField;
@property (nonatomic, strong) UITextField *serverHostField;
@property (nonatomic, strong) UITextField *serverPortField;
@property (nonatomic, strong) UITextField *threadCountField;
@property (nonatomic, strong) UIStepper *threadCountStepper;
@property (nonatomic, strong) UILabel *importStatusLabel;
@property (nonatomic, strong) UIButton *scanQRButton;
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
    self.endpointsExpanded = NO;
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
    self.segmentControl = [[UISegmentedControl alloc] initWithItems:@[ @"Inference", @"RMCluster Node", @"Logs" ]];
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

    self.logsContainer = [[UIView alloc] init];
    self.logsContainer.translatesAutoresizingMaskIntoConstraints = NO;
    self.logsContainer.hidden = YES;
    [self.view addSubview:self.logsContainer];

    self.logsViewController = [[RMLogsViewController alloc] init];
    [self addChildViewController:self.logsViewController];
    self.logsViewController.view.translatesAutoresizingMaskIntoConstraints = NO;
    [self.logsContainer addSubview:self.logsViewController.view];
    [self.logsViewController didMoveToParentViewController:self];

    [NSLayoutConstraint activateConstraints:@[
        [self.logsContainer.topAnchor constraintEqualToAnchor:self.segmentControl.bottomAnchor constant:12.0],
        [self.logsContainer.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.logsContainer.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.logsContainer.bottomAnchor constraintEqualToAnchor:bottomAnchor],
        [self.logsViewController.view.topAnchor constraintEqualToAnchor:self.logsContainer.topAnchor],
        [self.logsViewController.view.leadingAnchor constraintEqualToAnchor:self.logsContainer.leadingAnchor],
        [self.logsViewController.view.trailingAnchor constraintEqualToAnchor:self.logsContainer.trailingAnchor],
        [self.logsViewController.view.bottomAnchor constraintEqualToAnchor:self.logsContainer.bottomAnchor],
    ]];
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
    self.nicknameField = [self textFieldWithPlaceholder:@"Node Name (Optional)"];
    self.connectionStringField = [self textFieldWithPlaceholder:@"Paste rmcluster:// connection URL"];
    self.serverHostField = [self textFieldWithPlaceholder:@"Server IP or host"];
    self.serverPortField = [self textFieldWithPlaceholder:nil];
    self.serverPortField.keyboardType = UIKeyboardTypeNumberPad;
    self.threadCountField = [self numericField];
    [self.serverHostField addTarget:self action:@selector(coordinatorFieldsChanged:) forControlEvents:UIControlEventEditingChanged];
    [self.connectionStringField addTarget:self action:@selector(coordinatorFieldsChanged:) forControlEvents:UIControlEventEditingChanged];

    [self.rpcPane addArrangedSubview:self.nicknameField];

    self.endpointsStack = [self verticalStack];
    self.endpointsStack.hidden = YES;

    self.endpointsHeaderButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.endpointsHeaderButton.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeft;
    [self.endpointsHeaderButton setTitle:@"Endpoints" forState:UIControlStateNormal];
    self.endpointsHeaderButton.titleLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    [self.endpointsHeaderButton setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
    [self.endpointsHeaderButton addTarget:self action:@selector(toggleEndpointsSection) forControlEvents:UIControlEventTouchUpInside];

    UIStackView *endpointsHeaderRow = [[UIStackView alloc] init];
    endpointsHeaderRow.axis = UILayoutConstraintAxisHorizontal;
    endpointsHeaderRow.alignment = UIStackViewAlignmentCenter;
    endpointsHeaderRow.spacing = 8.0;
    [endpointsHeaderRow addArrangedSubview:self.endpointsHeaderButton];
    self.endpointsChevronLabel = [[UILabel alloc] init];
    self.endpointsChevronLabel.font = [UIFont boldSystemFontOfSize:15.0];
    self.endpointsChevronLabel.text = @">";
    [endpointsHeaderRow addArrangedSubview:self.endpointsChevronLabel];
    UIView *headerSpacer = [[UIView alloc] init];
    [headerSpacer setContentHuggingPriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisHorizontal];
    [endpointsHeaderRow addArrangedSubview:headerSpacer];
    [self.rpcPane addArrangedSubview:endpointsHeaderRow];
    [self.rpcPane addArrangedSubview:self.endpointsStack];

    UILabel *connectionTitle = [[UILabel alloc] init];
    connectionTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    connectionTitle.text = @"Connection";
    [self.rpcPane addArrangedSubview:connectionTitle];

    self.scanQRButton = [self listRowActionButtonWithTitle:@"Scan QR code"];
    [self.scanQRButton addTarget:self action:@selector(scanQRTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.rpcPane addArrangedSubview:self.scanQRButton];

    [self.rpcPane addArrangedSubview:self.connectionStringField];

    self.importStatusLabel = [[UILabel alloc] init];
    self.importStatusLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    self.importStatusLabel.numberOfLines = 0;
    self.importStatusLabel.textColor = [UIColor grayColor];
    [self.rpcPane addArrangedSubview:self.importStatusLabel];

    UILabel *coordinatorTitle = [[UILabel alloc] init];
    coordinatorTitle.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    coordinatorTitle.text = @"Coordinator";
    [self.rpcPane addArrangedSubview:coordinatorTitle];

    [self.rpcPane addArrangedSubview:self.serverHostField];
    [self.rpcPane addArrangedSubview:[self labeledFieldRowWithTitle:@"Server port" field:self.serverPortField]];

    self.threadCountStepper = [[UIStepper alloc] init];
    [self.threadCountStepper addTarget:self action:@selector(threadStepperChanged:) forControlEvents:UIControlEventValueChanged];
    [self.rpcPane addArrangedSubview:[self numericRowWithTitle:@"# Threads" field:self.threadCountField stepper:self.threadCountStepper]];

    UILabel *footer = [[UILabel alloc] init];
    footer.numberOfLines = 0;
    footer.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    footer.textColor = [UIColor grayColor];
    footer.text = @"Paste a rmcluster://connect URL, scan a QR code, or type the coordinator server IP, port, and token below.";
    [self.rpcPane addArrangedSubview:footer];

    self.rpcStatusLabel = [[UILabel alloc] init];
    self.rpcStatusLabel.numberOfLines = 0;
    self.rpcStatusLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
    [self.rpcPane addArrangedSubview:self.rpcStatusLabel];

    self.rpcStartStopButton = [self prominentActionButtonWithTitle:@"Connect to cluster" backgroundColor:[self rmSystemBlueColor]];
    [self.rpcStartStopButton addTarget:self action:@selector(rpcStartStopTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.rpcPane addArrangedSubview:self.rpcStartStopButton];
}

- (UIColor *)rmSystemBlueColor {
    return [UIColor colorWithRed:0.0 green:0.478 blue:1.0 alpha:1.0];
}

- (UIColor *)rmDestructiveRedColor {
    return [UIColor colorWithRed:1.0 green:0.231 blue:0.188 alpha:1.0];
}

- (UIColor *)rmWarningOrangeColor {
    return [UIColor colorWithRed:1.0 green:0.584 blue:0.0 alpha:1.0];
}

- (UIButton *)prominentActionButtonWithTitle:(NSString *)title backgroundColor:(UIColor *)backgroundColor {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    [button setTitle:title forState:UIControlStateNormal];
    button.titleLabel.font = [UIFont boldSystemFontOfSize:17.0];
    [button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    [button setTitleColor:[[UIColor whiteColor] colorWithAlphaComponent:0.6] forState:UIControlStateDisabled];
    button.backgroundColor = backgroundColor;
    button.layer.cornerRadius = 10.0;
    button.layer.masksToBounds = YES;
    button.contentHorizontalAlignment = UIControlContentHorizontalAlignmentCenter;
    button.contentEdgeInsets = UIEdgeInsetsMake(14.0, 16.0, 14.0, 16.0);
    button.translatesAutoresizingMaskIntoConstraints = NO;
    [button.heightAnchor constraintGreaterThanOrEqualToConstant:44.0].active = YES;
    return button;
}

- (UIButton *)listRowActionButtonWithTitle:(NSString *)title {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    [button setTitle:title forState:UIControlStateNormal];
    button.titleLabel.font = [UIFont systemFontOfSize:17.0];
    [button setTitleColor:[self rmSystemBlueColor] forState:UIControlStateNormal];
    button.backgroundColor = [UIColor colorWithWhite:0.96 alpha:1.0];
    button.layer.cornerRadius = 10.0;
    button.layer.borderWidth = 1.0 / [UIScreen mainScreen].scale;
    button.layer.borderColor = [UIColor colorWithWhite:0.82 alpha:1.0].CGColor;
    button.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeft;
    button.contentEdgeInsets = UIEdgeInsetsMake(12.0, 16.0, 12.0, 16.0);
    button.translatesAutoresizingMaskIntoConstraints = NO;
    [button.heightAnchor constraintGreaterThanOrEqualToConstant:44.0].active = YES;
    return button;
}

- (void)applyProminentButton:(UIButton *)button title:(NSString *)title backgroundColor:(UIColor *)backgroundColor enabled:(BOOL)enabled {
    [button setTitle:title forState:UIControlStateNormal];
    button.backgroundColor = backgroundColor;
    button.enabled = enabled;
    button.alpha = enabled ? 1.0 : 0.45;
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
    row.distribution = UIStackViewDistributionFillEqually;
    UILabel *label = [[UILabel alloc] init];
    label.text = title;
    label.adjustsFontSizeToFitWidth = YES;
    label.minimumScaleFactor = 0.8;
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
    NSString *coordinatorHost = self.settings.clusterServerHost;
    NSInteger coordinatorPort = self.settings.clusterServerPort;
    if (coordinatorPort == 0) {
        coordinatorPort = 4917;
    }
    self.nicknameField.text = self.settings.nickname ?: @"";
    self.serverHostField.text = coordinatorHost ?: @"";
    self.serverPortField.text = [NSString stringWithFormat:@"%ld", (long)coordinatorPort];
    self.threadCountField.text = [NSString stringWithFormat:@"%ld", (long)self.settings.threads];
    self.threadCountStepper.minimumValue = 1;
    self.threadCountStepper.maximumValue = 64;
    self.threadCountStepper.value = self.settings.threads;
    self.maxTokensValueLabel.text = @"200";
    [self updateTemperatureLabel];
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

    [self refreshEndpointsSection];
    self.rpcStatusLabel.text = self.service.rpcStatusMessage ?: @"";
    self.rpcStatusLabel.textColor = [UIColor grayColor];
    NSString *trimmedHost = [[self.serverHostField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    NSString *trimmedConnection = [[self.connectionStringField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    BOOL canConnect = trimmedHost.length > 0 || trimmedConnection.length > 0;

    switch (self.service.rpcServerState) {
        case RMRPCServerStateStarting:
        case RMRPCServerStateRecovering:
            self.rpcStatusLabel.textColor = [self rmWarningOrangeColor];
            [self applyProminentButton:self.rpcStartStopButton
                                 title:@"Cancel connection"
                       backgroundColor:[self rmWarningOrangeColor]
                               enabled:YES];
            break;
        case RMRPCServerStateRunning:
            self.rpcStatusLabel.textColor = [UIColor colorWithRed:0.2 green:0.78 blue:0.35 alpha:1.0];
            [self applyProminentButton:self.rpcStartStopButton
                                 title:@"Disconnect from cluster"
                       backgroundColor:[self rmDestructiveRedColor]
                               enabled:YES];
            break;
        case RMRPCServerStateDegraded:
        case RMRPCServerStateUnavailable:
        default:
            self.rpcStatusLabel.textColor = [UIColor grayColor];
            [self applyProminentButton:self.rpcStartStopButton
                                 title:@"Connect to cluster"
                       backgroundColor:[self rmSystemBlueColor]
                               enabled:canConnect];
            break;
    }

    BOOL fieldsEditable = (self.service.rpcServerState == RMRPCServerStateIdle
                           || self.service.rpcServerState == RMRPCServerStateUnavailable
                           || self.service.rpcServerState == RMRPCServerStateDegraded);
    self.nicknameField.enabled = fieldsEditable;
    self.serverHostField.enabled = fieldsEditable;
    self.serverPortField.enabled = fieldsEditable;
    self.threadCountField.enabled = fieldsEditable;
    self.threadCountStepper.enabled = fieldsEditable;
    self.connectionStringField.enabled = fieldsEditable;
    self.scanQRButton.enabled = fieldsEditable;
}

- (void)toggleEndpointsSection {
    self.endpointsExpanded = !self.endpointsExpanded;
    [self refreshEndpointsSection];
}

- (void)refreshEndpointsSection {
    self.endpointsStack.hidden = !self.endpointsExpanded;

    for (UIView *view in self.endpointsStack.arrangedSubviews) {
        [self.endpointsStack removeArrangedSubview:view];
        [view removeFromSuperview];
    }

    self.endpointsChevronLabel.transform = self.endpointsExpanded
        ? CGAffineTransformMakeRotation((CGFloat)(M_PI_2))
        : CGAffineTransformIdentity;

    if (!self.endpointsExpanded) {
        return;
    }

    NSArray<RMLocalInterface *> *interfaces = [RMInferenceService allLocalIPv4Interfaces];
    if (interfaces.count == 0) {
        UILabel *emptyLabel = [[UILabel alloc] init];
        emptyLabel.text = @"No network interfaces found";
        emptyLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
        emptyLabel.textColor = [UIColor grayColor];
        [self.endpointsStack addArrangedSubview:emptyLabel];
        return;
    }

    BOOL isRunning = (self.service.rpcServerState == RMRPCServerStateRunning
        || self.service.rpcServerState == RMRPCServerStateStarting);
    NSInteger listenPort = [RMRpcSettings listenPort];

    for (RMLocalInterface *interface in interfaces) {
        UIStackView *row = [[UIStackView alloc] init];
        row.axis = UILayoutConstraintAxisHorizontal;
        row.spacing = 12.0;
        row.alignment = UIStackViewAlignmentTop;

        UIView *statusDot = [[UIView alloc] init];
        statusDot.translatesAutoresizingMaskIntoConstraints = NO;
        statusDot.backgroundColor = isRunning
            ? [UIColor colorWithRed:0.2 green:0.78 blue:0.35 alpha:1.0]
            : [UIColor colorWithWhite:0.75 alpha:1.0];
        statusDot.layer.cornerRadius = 4.5;
        [statusDot.widthAnchor constraintEqualToConstant:9.0].active = YES;
        [statusDot.heightAnchor constraintEqualToConstant:9.0].active = YES;

        UIStackView *textColumn = [self verticalStack];
        textColumn.spacing = 2.0;
        UILabel *rpcLabel = [[UILabel alloc] init];
        rpcLabel.font = [UIFont fontWithName:@"Menlo-Bold" size:15.0] ?: [UIFont boldSystemFontOfSize:15.0];
        rpcLabel.text = [NSString stringWithFormat:@"RPC %@", interface.ip];
        UILabel *storageLabel = [[UILabel alloc] init];
        storageLabel.font = [UIFont fontWithName:@"Menlo-Regular" size:12.0] ?: [UIFont systemFontOfSize:12.0];
        storageLabel.textColor = [UIColor grayColor];
        storageLabel.text = [NSString stringWithFormat:@"Storage %@", interface.ip];
        UILabel *ifaceLabel = [[UILabel alloc] init];
        ifaceLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleCaption1];
        ifaceLabel.textColor = [UIColor grayColor];
        ifaceLabel.text = interface.label;
        [textColumn addArrangedSubview:rpcLabel];
        [textColumn addArrangedSubview:storageLabel];
        if (interface.label.length > 0) {
            [textColumn addArrangedSubview:ifaceLabel];
        }

        UIButton *copyButton = [UIButton buttonWithType:UIButtonTypeSystem];
        [copyButton setTitle:@"Copy" forState:UIControlStateNormal];
        copyButton.titleLabel.font = [UIFont systemFontOfSize:13.0];
        NSString *copyValue = [NSString stringWithFormat:@"%@:%ld", interface.ip, (long)listenPort];
        objc_setAssociatedObject(copyButton, @"copyValue", copyValue, OBJC_ASSOCIATION_COPY_NONATOMIC);
        [copyButton addTarget:self action:@selector(copyEndpointTapped:) forControlEvents:UIControlEventTouchUpInside];

        [row addArrangedSubview:statusDot];
        [row addArrangedSubview:textColumn];
        [row addArrangedSubview:copyButton];
        [self.endpointsStack addArrangedSubview:row];
    }
}

- (void)copyEndpointTapped:(UIButton *)sender {
    NSString *value = objc_getAssociatedObject(sender, @"copyValue");
    if (value.length > 0) {
        [UIPasteboard generalPasteboard].string = value;
        self.importStatusLabel.text = @"Copied RPC endpoint to clipboard.";
    }
}

- (void)updateTemperatureLabel {
    self.temperatureValueLabel.text = [NSString stringWithFormat:@"Temperature  %.2f", self.temperatureSlider.value];
}

- (void)segmentChanged:(UISegmentedControl *)sender {
    BOOL showingInference = sender.selectedSegmentIndex == 0;
    BOOL showingRPC = sender.selectedSegmentIndex == 1;
    BOOL showingLogs = sender.selectedSegmentIndex == 2;
    self.inferencePane.hidden = !showingInference;
    self.rpcPane.hidden = !showingRPC;
    self.scrollView.hidden = showingLogs;
    self.logsContainer.hidden = !showingLogs;
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
    if (self.service.rpcServerState == RMRPCServerStateRunning || self.service.rpcServerState == RMRPCServerStateStarting) {
        [self.service stopRPCServer];
        return;
    }

    if (![self prepareCoordinatorSettingsForStart]) {
        return;
    }

    [self syncSettingsFromFields];
    [self.settings persistClusterConnection];
    [self.service startRPCServerWithCoordinatorHost:self.settings.clusterServerHost
                                    coordinatorPort:self.settings.clusterServerPort
                                           nickname:self.settings.nickname
                                            threads:self.settings.threads
                                           deviceId:self.settings.deviceId];
}

- (BOOL)prepareCoordinatorSettingsForStart {
    NSString *trimmedHost = [[self.serverHostField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (trimmedHost.length > 0) {
        self.settings.clusterServerHost = trimmedHost;
        NSInteger port = [self.serverPortField.text integerValue];
        self.settings.clusterServerPort = port > 0 ? port : [RMRpcSettings defaultClusterServerPort];
        self.importStatusLabel.text = @"";
        return YES;
    }

    NSString *pendingConnection = [[self.connectionStringField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (pendingConnection.length > 0) {
        if (![self applyConnectionConfigFromString:pendingConnection]) {
            return NO;
        }
        return YES;
    }

    self.importStatusLabel.text = @"Enter a coordinator server IP or paste a connection string.";
    return NO;
}

- (void)syncSettingsFromFields {
    self.settings.nickname = self.nicknameField.text ?: @"";
    self.settings.clusterServerHost = self.serverHostField.text ?: @"";
    NSInteger port = [self.serverPortField.text integerValue];
    self.settings.clusterServerPort = port > 0 ? port : [RMRpcSettings defaultClusterServerPort];
    self.settings.threads = MAX(1, [self.threadCountField.text integerValue]);
}

- (BOOL)applyConnectionConfigFromString:(NSString *)rawValue {
    RMConnectionBootstrapPayload *payload = [RMConnectionBootstrapPayload payloadWithRawValue:rawValue];
    if (payload == nil) {
        self.importStatusLabel.text = @"Could not parse connection data.";
        return NO;
    }

    self.connectionStringField.text = [rawValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    self.settings.clusterServerHost = payload.host ?: @"";
    self.serverHostField.text = self.settings.clusterServerHost;

    if (payload.port != nil) {
        self.settings.clusterServerPort = payload.port.integerValue;
        self.serverPortField.text = [NSString stringWithFormat:@"%ld", (long)self.settings.clusterServerPort];
    }

    if (payload.device.length > 0) {
        self.settings.nickname = payload.device;
        self.nicknameField.text = payload.device;
    }

    self.segmentControl.selectedSegmentIndex = 1;
    [self segmentChanged:self.segmentControl];
    self.importStatusLabel.text = @"";
    return YES;
}

- (void)coordinatorFieldsChanged:(id)sender {
    if (self.service.rpcServerState == RMRPCServerStateIdle
        || self.service.rpcServerState == RMRPCServerStateUnavailable
        || self.service.rpcServerState == RMRPCServerStateDegraded) {
        NSString *trimmedHost = [[self.serverHostField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
        NSString *trimmedConnection = [[self.connectionStringField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
        BOOL canConnect = trimmedHost.length > 0 || trimmedConnection.length > 0;
        [self applyProminentButton:self.rpcStartStopButton
                             title:@"Connect to cluster"
                   backgroundColor:[self rmSystemBlueColor]
                           enabled:canConnect];
    }
}

- (void)threadStepperChanged:(UIStepper *)sender {
    self.threadCountField.text = [NSString stringWithFormat:@"%ld", (long)sender.value];
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
