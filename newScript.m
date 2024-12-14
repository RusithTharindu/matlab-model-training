users = ["U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08", "U09", "U10"];
target_user = "U01";

allTrainData = [];
allTrainLabels = [];
allTestData = [];
allTestLabels = [];

epochs = 1000;
target_user_train_count = 0.7;
target_user_test_count = 0.3;
imposter_user_train_count = 0.7;
imposter_user_test_count = 0.3;

for targetUser = 1:10
    train_data = [];
    test_data = [];

    for user = 1:10
        freqDomainFDay = load(['dataset/', char(users(user)), '_Acc_FreqD_FDay']);
        acc_freq_fday = freqDomainFDay.Acc_FD_Feat_Vec;

        freqDomainMDay = load(['dataset/', char(users(u)), '_Acc_FreqD_MDay']);
        acc_freq_mday = freqDomainMDay.Acc_FD_Feat_Vec;

        timeDomainFDay = load(['dataset/', char(users(u)), '_Acc_TimeD_FDay']);
        acc_time_fday = timeDomainFDay.Acc_TD_Feat_Vec;

        timeDomainMDay = load(['dataset/', char(users(u)), '_Acc_TimeD_MDay']);
        acc_time_mday = timeDomainMDay.Acc_TD_Feat_Vec;

        combine_train_data = acc_freq_fday + acc_time_fday;
        train_data = [train_data;combine_train_data];

        combine_test_data = acc_freq_mday + acc_time_mday;
        test_data = [test_data; combine_test_data];

        if(user == targetUser)
            target_samples = size(train_data,1);
            target_indices = randperm(target_samples);
            targetTrain = round(target_user_train_count * target_samples);

            targetTrainData = train_data(target_indices(1:targetTrain), :);
            targetTrainLabels = ones(size(targetTrainData, 1), 1);

            allTrainData = [allTrainData; targetTrainData];
            allTrainLabels = [allTrainLabels; targetTrainLabels];

            numTargetTest = round(0.3)
            

        end

    end

end