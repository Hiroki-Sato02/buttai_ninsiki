% urllist.txtに書かれているリンクの画像を保存する
function img_dloader()

    % テキストファイルの読み込み
    list=textread('urllist.txt','%s');
    
    OUTDIR='p2_test'; % 保存先ディレクトリ指定
    for i=1:size(list,1)
      fname=strcat(OUTDIR,'/',num2str(i,'%04d'),'.jpg')
      websave(fname, list{i});
    end
    
end