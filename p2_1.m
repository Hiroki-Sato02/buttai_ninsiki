% レポート課題２
function p2_1()
    net = alexnet;
    n_train = 25; %25 or 50
    n_test = 300;
    n_bgimg = 1142;

    % 作成 ----------------------------------------------------------
    list={}; PosList={}; test={};
    LIST={'p2_train' 'p2_test' 'bgimg'};
    for i=1:length(LIST)
        DIR=strcat(LIST(i),'/');
        W=dir(DIR{:});
        for j=1:size(W)
            if (strfind(W(j).name,'.jpg'))
                fn=strcat(DIR{:},W(j).name);
                if i==1
                    if j<=27 % 50の場合はコメントアウト
                        PosList={PosList{:} fn};
                    end
                elseif i==2
                    test={test{:} fn}; 
                else
                    list={list{:} fn};
                end
            end
        end
    end

    sel=randperm(1142,500); % 1~1142の中で500個の整数乱数生成
    
    PosList = PosList';
    test = test';
    NegList=list(sel)'; % 500枚
    train={PosList{:} NegList{:}};
    train_label = [ones(numel(PosList),1); ones(numel(NegList),1)*(-1)];
    
    % 学習 ---------------------------------------------------------------
    n = 500+n_train;
    dcnnf_train_list=zeros(n,4096);
    for j=1:n
        img = imread(train{j});
        reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
        dcnnf = activations(net,reimg,'fc7');
        dcnnf = squeeze(dcnnf);
        dcnnf = dcnnf/norm(dcnnf);
        dcnnf_train_list(j, :) = dcnnf';
    end

    n = 300;
    dcnnf_eval_list=zeros(n,4096);
    for i=1:n
        img = imread(test{i});
        reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
        dcnnf = activations(net,reimg,'fc7');
        dcnnf = squeeze(dcnnf);
        dcnnf = dcnnf/norm(dcnnf);
        dcnnf_eval_list(i, :) = dcnnf';
    end

    % 分類
    data = dcnnf_train_list;
    training_data = repmat(sqrt(abs(data)).*sign(data),[1 3]).*[0.8*ones(size(data)) 0.6*cos(0.6*log(abs(data)+eps)) 0.6*sin(0.6*log(abs(data)+eps))];

    data = dcnnf_eval_list;
    testing_data = repmat(sqrt(abs(data)).*sign(data),[1 3]).*[0.8*ones(size(data)) 0.6*cos(0.6*log(abs(data)+eps)) 0.6*sin(0.6*log(abs(data)+eps))];
    
    % 学習関数fitcsvm (linear(線形)カーネル) 
    model = fitcsvm(training_data, train_label,'KernelFunction','linear'); 
   
   % 分類関数svmpredict
   [predicted_label, score] = predict(model, testing_data);

   % 降順 ('descent') でソートして，ソートした値とソートインデックスを取得します．
   [sorted_score,sorted_idx] = sort(score(:,2),'descend');
    
   % ランキング出力
   for i=1:numel(sorted_idx)
        fprintf('%s %f\n',test{sorted_idx(i)},sorted_score(i));
   end

end