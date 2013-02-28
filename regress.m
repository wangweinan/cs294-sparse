feature(’numThreads’, getenv(NSLOTS))
% Sentiment Scorer with Linear Regression

% This is tokenized data from the Multi-Domain Sentiment Dataset found at:
% http://www.cs.jhu.edu/~mdredze/datasets/sentiment/
% The mat file contains three variables:
% - tokens
% - scnt
% - smap
%load('data/tokenized.mat')

load('tokens.mat','tokens');
%load('tokenstest.mat', 'tokenstest');
%load('scnt.mat', 'scnt');
load('smap.mat', 'smap');

load('stopwords.mat', 'stopWordIndexes');
load('stemmedSmap.mat', 'smapUnique', 'uniqToSmap',...
     'smapToUniq', 'stemmedSmap');
lenTokens = length(tokens);
dictSize = length(smap);

%!!! get a subset of tokens!!!%
%tokens = tokens(1:100000);
% tokens is a 3xN matrix, but the first two rows are useless.
% Get rid of the first two rows.
% This should reduce the amount of memory required significantly.
%tokens = tokens(3, :);

% Find the token index for the common tokens.
RATING_BEGIN = find(ismember(smap, '<rating>'));
REVIEW_BEGIN = find(ismember(smap, '<review>'));
REVIEW_END = find(ismember(smap,'</review>'));

% Extract rating positions and find the total number of reviews.
reviewTextBeginPositions = find(tokens == REVIEW_BEGIN);
reviewTextEndPositions = find(tokens == REVIEW_END);
numReviews = min(length(reviewTextEndPositions),length(reviewTextBeginPositions));
dictStemmedSize = length(uniqToSmap);
%Xdefault = sparse(1 + dictSize, numReviews);
%XnoStopWord = sparse(1 + dictSize, numReviews);
Xstemmed = sparse(1 + dictStemmedSize, numReviews);

map = containers.Map('KeyType', 'char', 'ValueType', 'logical');
uniqueReviews = zeros(numReviews,1);
uniqueReviewsIndex = 1;

for i = 1 : numReviews
    
    % report progress
    %if mod(i, 10000) == 0
       % i;
      %  length(uniqueReviews)
    %end
    
    reviewTexts = tokens(reviewTextBeginPositions(i) : ...
                         reviewTextEndPositions(i));
    
    % skip the review if it has been observed before.
    hashkey = mat2str(reviewTexts(1 : min(10, length(reviewTexts))));
    if ~isKey(map, hashkey)
        map(hashkey) = true;
        uniqueReviews(uniqueReviewsIndex) = i;
        uniqueReviewsIndex = uniqueReviewsIndex +1;
        
        reviewTextsMod = reviewTexts(reviewTexts<=dictSize);
        
        % remove stop words
        reviewNoStopWords = reviewTextsMod(ismember(reviewTextsMod, stopWordIndexes)==0);
        
        % stemming
        %textLen = length(reviewTextsMod);
        textLen = length(reviewNoStopWords);
        reviewStemmed = zeros(textLen,1);
        for j = 1 : textLen
            reviewStemmed(j) = smapToUniq(reviewNoStopWords(j));
        end
        
        %Xdefault(:, i) = [1; sparse(double(reviewTextsMod), 1, 1, ...
        %    dictSize, 1)];
        Xstemmed(:, i) = [1; sparse(double(reviewStemmed), 1, 1, ...
            dictStemmedSize, 1)];
        %Xstopwords(:, i) = [1; sparse(double(reviewStopWords), 1, 1, ...
        %    dictSize, 1)];
    end
end


uniqueReviews = uniqueReviews(uniqueReviews>0);

% process y.
% Extract ratings (assume all ratings are integers).
ratingPositions = find(tokens == RATING_BEGIN);
y = cell2mat(smap(tokens(ratingPositions + 1))) - '0';
yuniq = y(uniqueReviews);


%Xuniq = Xdefault(:, uniqueReviews);
%save('model-default.mat', 'Xuniq', 'yuniq');

Xuniq = Xstemmed(:, uniqueReviews);
save('model-stemmed.mat', 'Xuniq', 'yuniq');
%Xuniq = Xstopwords(:, uniqueReviews);
%save('model-stopwords.mat', 'Xuniq', 'yuniq');




