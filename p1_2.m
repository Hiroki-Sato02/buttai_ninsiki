% レポート課題１のBoFベクトルと非線形SVMによる分類のプログラム
function p1_2()

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
    Training={PosList{:} NegList{:}};

    % CoodBookの生成 --------------------------------------------------
    Features=[];
    for i=1:600
      I=rgb2gray(imread(Training{i}));
      %p=detectSURFFeatures(I);
      p =createRandomPoints(I, 1000);
      [f,p2]=extractFeatures(I,p);
      Features=[Features; f];
    end

    if (size(Features,1) > 50000)
        Features=Features(randperm(size(Features,1),50000), :);
    end
    [idx,CODEBOOK]=kmeans(Features,500);

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

    
        % 学習 (bofベクトル)
        n = 480;
        bof_train = zeros(n,1000);
        for j=1:n  % 各画像についての for-loop
           % j番目の画像読み込み
           img = imread(train{j});
           % SURF特徴抽出
           I=rgb2gray(img);
           %p=detectSURFFeatures(I);
           p =createRandomPoints(I, 1000);
           [f,p2]=extractFeatures(I,p);
           min = 10000;
           index = 1;
           for q=1:size(p2,1)  % 各特徴点についての for-loop
              % 一番近いcodebook中のベクトルを探してindexを求める．
              a = f(q, :);
              for p=1:500
                 b = CODEBOOK(p, :);
                 d = (a-b).^2;
                 d=sqrt(sum(d'));
                 if min > d
                    min = d;
                    index = p;
                 end
              end
              % bofヒストグラム行列のj番目の画像のindexに投票
              bof_train(j,index)=bof_train(j,index)+1;
           end
        end
        % sum(A,2)で行ごとの合計を求めて，それを各行の要素について割ることによって，各行の合計値を１として正規化する．
        bof_train = bof_train ./ sum(bof_train,2);

        % 評価 (bofベクトル)
        n = 120;
        bof_eval = zeros(n,1000);
        for j=1:n  % 各画像についての for-loop
           % j番目の画像読み込み
           img = imread(eval{j});
           % SURF特徴抽出
           I=rgb2gray(img);
           %p=detectSURFFeatures(I);
           p =createRandomPoints(I, 1000);
           [f,p2]=extractFeatures(I,p);
           min = 10000;
           index = 1;
           for q=1:size(p2,1)  % 各特徴点についての for-loop
              % 一番近いcodebook中のベクトルを探してindexを求める．
              a = f(q, :);
              for p=1:500
                 b = CODEBOOK(p, :);
                 d = (a-b).^2;
                 d=sqrt(sum(d'));
                 if min > d
                    min = d;
                    index = p;
                 end
              end
              % bofヒストグラム行列のj番目の画像のindexに投票
              bof_eval(j,index)=bof_eval(j,index)+1;
           end
        end
        % sum(A,2)で行ごとの合計を求めて，それを各行の要素について割ることによって，各行の合計値を１として正規化する．
        bof_eval = bof_eval ./ sum(bof_eval,2);

        % 学習関数fitcsvm (RBF(非線形)カーネル) 
        model = fitcsvm(bof_train, train_label,'KernelFunction','rbf', 'KernelScale','auto');
   
        % 分類関数svmpredict
        [predicted_label, scores] = predict(model, bof_eval);

        ac = numel(find(eval_label==predicted_label))/numel(eval_label); % 評価(認識精度値を出力)
        accuracy = [accuracy ac];

    end

    fprintf('accuracy: %f\n',mean(accuracy))
    
end

function PT=createRandomPoints(I,num)
  [sy sx]=size(I);
  sz=[sx sy];
  for i=1:num
    s=0;
    while s<1.6
      s=randn()*3+3;
    end
    p=ceil((sz-ceil(s)*2).*rand(1,2)+ceil(s));
    if i==1
      PT=[SURFPoints(p,'Scale',s)];
    else
      PT=[PT; SURFPoints(p,'Scale',s)];
    end
  end
end