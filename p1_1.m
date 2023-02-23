% レポート課題１のカラーヒストグラムと最近傍分類による分類のプログラム
function p1_1()

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

    
        % 学習 (カラーヒストグラム)
        database=[];
        for j=1:length(train)
            X=imread(train{j});
            % ヒストグラムの生成
            RED=X(:,:,1); GREEN=X(:,:,2); BLUE=X(:,:,3);
            X64=floor(double(RED)/64) *4*4 + floor(double(GREEN)/64) *4 + floor(double(BLUE)/64);
            X64_vec=reshape(X64,1,numel(X64));
            h=histc(X64_vec,[0:63]);
            h = h / sum(h); % 要素の合計が１になるように正規化
            database=[database; h];
        end
        

        % 分類 (カラーヒストグラム同士の最近傍分類(ユークリッド距離))
        ac = 0;
        for j=1:length(eval)
            X=imread(eval{j});
            % ヒストグラムの生成
            RED=X(:,:,1); GREEN=X(:,:,2); BLUE=X(:,:,3);
            X64=floor(double(RED)/64) *4*4 + floor(double(GREEN)/64) *4 + floor(double(BLUE)/64);
            X64_vec=reshape(X64,1,numel(X64));
            h=histc(X64_vec,[0:63]);
            h = h / sum(h); % 要素の合計が１になるように正規化
            % hとdatabaseのユークリッド距離を計算して一番近いindexが同じpositiveもしくはnegativeだったら+1
            dist_min=99999; % 最小距離
            dist_min_idx=1; % 最小距離のindex 
            for q=1:length(database)
                b=(h-database(q)).^2;
                c=sqrt(sum(b'));
                if dist_min > c
                    dist_min = c;
                    dist_min_idx = q;
                end
            end
            if eval_label(j) == train_label(dist_min_idx) % labelが同じなら正しく分類されている
                ac = ac + 1;
            end
        end

        ac = ac/numel(eval); % 評価(認識精度値を出力)
        accuracy = [accuracy ac];
    end

    fprintf('accuracy: %f\n',mean(accuracy))
    
end
