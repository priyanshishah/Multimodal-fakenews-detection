s=importdata('test_actual.txt');
documents = preprocessText(s);
% documents(1:5)
bag = bagOfWords(documents)
bag = removeInfrequentWords(bag,2);
bag = removeEmptyDocuments(bag)
numTopics = 7;
mdl = fitlda(bag,numTopics,'Verbose',0);
figure;
for topicIdx = 1:2
    subplot(2,2,topicIdx)
    wordcloud(mdl,topicIdx);
    title("rumor " + topicIdx)
end
newDocument = tokenizedDocument("rumor");
topicMixture = transform(mdl,newDocument);
% figure
% bar(topicMixture)
% xlabel("Topic Index")
% ylabel("Probability")
% title("Document Topic Probabilities")
% figure
% topicMixtures = transform(mdl,documents(1:5));
% barh(topicMixtures(1:5,:),'stacked')
% xlim([0 1])
% title("Topic Mixtures")
% xlabel("Topic Probability")
% ylabel("Document")
% legend("Topic " + string(1:numTopics),'Location','northeastoutside')
% bar(topicMixture)
% xlabel("Topic Index")
% ylabel("Probability")
% title("Document Topic Probabilities")
% figure
% topicMixtures = transform(mdl,documents(1:2));
% barh(topicMixtures(1:2,:),'stacked')
% xlim([0 1])
% title("Mixtures")
% xlabel(" Probability")
% ylabel("Document")
% legend("Topic " + string(1:numTopics),'Location','northeastoutside')

