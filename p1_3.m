% レポート課題１のAlexNetやVGG16などによるDCNN特徴量と線形SVMによる分類のプログラム
function p1_3()
    net = alexnet;

    % 作成 ----------------------------------------------------------
    list={};
    LIST={'imgdir_ramen' 'bgimg'};
    for i=1:length(LIST)
        DIR=strcat(LIST(i),'/');
        W=dir(DIR{:});
        for j=1:size(W)
            if (strfind(W(j).name,'.jpg'))
                fn=strcat(DIR{:},W(j).name);
                list={list{:} fn};

            end
        end
    end

    sel=randperm(942,400)+200; % 200~1142の中で400個の整数乱数生成
    
    PosList=list(1:200)'; % 200枚
    NegList=list(sel)'; % 400枚

    % 5-fold cross validation --------------------------------------------------
    cv=5;
    idx_pos=[1:200];
    idx_neg=[1:400];
    accuracy=[];
    
    % idx番目(idxはcvで割った時の余りがi-1)が評価データ
    % それ以外は学習データ
    for i=1:cv 
        train_pos = PosList(find(mod(idx_pos,cv)~=(i-1)),:);
        eval_pos = PosList(find(mod(idx_pos,cv)==(i-1)),:);
        train_neg = NegList(find(mod(idx_neg,cv)~=(i-1)),:);
        eval_neg = NegList(find(mod(idx_neg,cv)==(i-1)),:);
    
        train = [train_pos; train_neg];
        eval = [eval_pos; eval_neg];
    
        train_label = [ones(numel(train_pos),1); ones(numel(train_neg),1)*(-1)];
        eval_label = [ones(numel(eval_pos),1); ones(numel(eval_neg),1)*(-1)];

    
        % 学習
        n = 480;
        dcnnf_train_list=zeros(n,4096);
        for j=1:n
            img = imread(train{j});
            reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
            dcnnf = activations(net,reimg,'fc7');
            dcnnf = squeeze(dcnnf);
            dcnnf = dcnnf/norm(dcnnf);
            dcnnf_train_list(j, :) = dcnnf';
        end
        n = 120;
        dcnnf_eval_list=zeros(n,4096);
        for j=1:n
            img = imread(eval{j});
            reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
            dcnnf = activations(net,reimg,'fc7');
            dcnnf = squeeze(dcnnf);
            dcnnf = dcnnf/norm(dcnnf);
            dcnnf_eval_list(j, :) = dcnnf';
        end

        % 分類 (分類用のmファイルを作りましょう)
        data = dcnnf_train_list;
        training_data = repmat(sqrt(abs(data)).*sign(data),[1 3]).*[0.8*ones(size(data)) 0.6*cos(0.6*log(abs(data)+eps)) 0.6*sin(0.6*log(abs(data)+eps))];
      
        data = dcnnf_eval_list;
        testing_data = repmat(sqrt(abs(data)).*sign(data),[1 3]).*[0.8*ones(size(data)) 0.6*cos(0.6*log(abs(data)+eps)) 0.6*sin(0.6*log(abs(data)+eps))];

        % 学習関数fitcsvm (linear(線形)カーネル) 
        model = fitcsvm(training_data, train_label,'KernelFunction','linear'); 
   
        % 分類関数svmpredict
        [predicted_label, scores] = predict(model, testing_data);

        ac = numel(find(eval_label==predicted_label))/numel(eval_label); % 評価(認識精度値を出力)
        accuracy = [accuracy ac];

    end

    fprintf('accuracy: %f\n',mean(accuracy))
    
end
