feature('numThreads', 12)
% Sentiment Scorer with Linear Regression

% This is tokenized data from the Multi-Domain Sentiment Dataset found at:
% http://www.cs.jhu.edu/~mdredze/datasets/sentiment/
% The mat file contains three variables:
% - tokens
% - scnt
% - smap
%load('data/tokenized.mat')

load('tokens.mat','tokens');
load('smap.mat', 'smap');
load('stopwords.mat', 'stopWordIndexes');
load('stemmedSmap.mat', 'smapUnique', 'uniqToSmap',...
     'smapToUniq', 'stemmedSmap');

dictSize = length(smap);

% Find the token index for the common tokens.
RATING_BEGIN = find(ismember(smap, '<rating>'));
REVIEW_BEGIN = find(ismember(smap, '<review_text>'));
REVIEW_END = find(ismember(smap,'</review_text>'));

% Extract rating positions and find the total number of reviews.
reviewTextBeginPositions = find(tokens == REVIEW_BEGIN);
reviewTextEndPositions = find(tokens == REVIEW_END);
numReviews = min(length(reviewTextEndPositions),length(reviewTextBeginPositions));
dictStemmedSize = length(uniqToSmap);


map = containers.Map('KeyType', 'char', 'ValueType', 'logical');
uniqueReviews = zeros(numReviews,1);
uniqueReviewsIndex = 1;
for i = 1 : numReviews
    
    reviewTexts = tokens(reviewTextBeginPositions(i) : ...
                         reviewTextEndPositions(i));
    
    % skip the review if it has been observed before.
    hashkey = mat2str(reviewTexts(1 : min(10, length(reviewTexts))));
    if ~isKey(map, hashkey)
        map(hashkey) = true;
        uniqueReviews(uniqueReviewsIndex) = i;
        uniqueReviewsIndex = uniqueReviewsIndex +1;
    end
end
uniqueReviews = uniqueReviews(uniqueReviews>0);
numUniqueReviews = length(uniqueReviews);
Xstemmed = sparse(1 + dictStemmedSize, numUniqueReviews);


parfor i = 1 : numUniqueReviews
    m=uniqueReviews(i);
    reviewTexts = tokens(reviewTextBeginPositions(m) : ...
                         reviewTextEndPositions(m));
    
        
        reviewTextsMod = reviewTexts(reviewTexts<=dictSize);
        
        % remove stop words
        reviewNoStopWords = reviewTextsMod(ismember(reviewTextsMod, stopWordIndexes)==0);
        
        % stemming
        
        textLen = length(reviewNoStopWords);
        reviewStemmed = zeros(textLen,1);
        for j = 1 : textLen
            reviewStemmed(j) = smapToUniq(reviewNoStopWords(j));
        end
        
       
        Xstemmed(:, i) = [1; sparse(double(reviewStemmed), 1, 1, ...
            dictStemmedSize, 1)];
       
end




% process y.
% Extract ratings (assume all ratings are integers).
ratingPositions = find(tokens == RATING_BEGIN);
y = cell2mat(smap(tokens(ratingPositions + 1))) - '0';
yuniq = y(uniqueReviews);


save('model-stemmed.mat', 'XStemmed', 'yuniq');





