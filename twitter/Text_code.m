clc
clear all
close all
u=fileread('actualdata.txt');
u(1,:);
u1=string(u);
u1=splitlines(u1);
TF = (u1 == ' ');
u1(TF) = [];
u1(1:10);
p = [".","?","!",",",";",":",' ','$','/','.','-',':','&','*', ...          % remove those
    '+','=','[',']','?','!','(',')','{','}',',', ...
    '"','>','_','<',';','%'];
u2 = replace(u1,p," ");
u2(1,:)
u3= strip(u2);
u2(1,:)
sonnetWords = strings(0);
for i = 1:length(u3)
sonnetWords = [sonnetWords ; split(u3(i))];
end
sonnetWords(1:10)
sonnetWords = lower(sonnetWords);
[words,~,idx] = unique(sonnetWords);
numOccurrences = histcounts(idx,numel(words));
[rankOfOccurrences,rankIndex] = sort(numOccurrences,'descend');
wordsByFrequency = words(rankIndex);
loglog(rankOfOccurrences);
xlabel('Rank of word (most to least common)');
ylabel('Number of Occurrences');
wordsByFrequency;
numOccurrences = numOccurrences(rankIndex);
numOccurrences = numOccurrences';
numWords = length(sonnetWords);
T = table;
T.Words = wordsByFrequency;
T.NumOccurrences = numOccurrences;
T.PercentOfText = numOccurrences / numWords * 100.0;
T.CumulativePercentOfText = cumsum(numOccurrences) / numWords * 100.0;
T(1:26616,:);
value1=fileread('test_actual.txt');
value1(1,:);
value2=string(value1);
value3=splitlines(value1);
value_TF = (value1 == "");
value1(value_TF) = [];
%  value1(1:10)
%     'Delimiter','\t','ReadVariableNames',0);
% AFINN.Properties.VariableNames = {'Term','Score'};          % add var names
% AFINN.Properties.VariableNames = {'Term'}; 
% stopwordsURL ='http://www.textfixer.com/resources/common-english-words.txt';
% stopWords = webread(stopwordsURL); 
% stopWords = importdata('data.txt');
% read stop words
stopWords = importdata('train_rumor.txt');
% stopWords = importdata('train_rumor.txt');
% stopWords = importdata('train_nonrumor.txt');
% stopWords = importdata('test_rumor.txt');
% stopWords = importdata('test_nonrumor.txt');
% str=[];
% stopWords = split(string(stopWords),',');   
%  for i = 1:length(stopWords)
%      str = [str; split(stopWords{i}, ",")];
%  end% split stop words
tweetscnt=1000;
delimiters = {' ','$','/','.','-',':','&','*', ...          % remove those
    '+','=','[',']','?','!','(',')','{','}',',', ...
    '"','>','_','<',';','%',char(10),char(13)};
% tokens = cell(fake_news.tweetscnt,1);                       % cell arrray as accumulator
% expUrls = strings(fake_news.tweetscnt,1);                   % cell arrray as accumulator
% dispUrls = strings(fake_news.tweetscnt,1);                  % cell arrray as accumulator
% scores = zeros(fake_news.tweetscnt,1);                      % initialize accumulator
tokens = cell(tweetscnt,1);                       % cell arrray as accumulator
expUrls = strings(tweetscnt,1);                   % cell arrray as accumulator
dispUrls = strings(tweetscnt,1);                  % cell arrray as accumulator
scores = zeros(tweetscnt,1); 
for ii = 1:tweetscnt                              % loop over tweets
%     tweet = string(fake_news.statuses(ii).status.text);     % get tweet
     tweet = value2;
    s = split(tweet, delimiters)';                          % split tweet by delimiters
    s = lower(s);                                           % use lowercase
    s = regexprep(s, '[0-9]+','');                          % remove numbers
    s = regexprep(s,'(http|https)://[^\s]*','');            % remove urls
    s = erase(s,'''s');                                     % remove possessive s
    s(s == '') = [];                                        % remove empty strings
    s(ismember(s, stopWords)) = [];                         % remove stop words
    tokens{ii} = s;                                         % add to the accumulator
%     scores(ii) = sum(AFINN.Score(ismember(AFINN.Term,s)));  % add to the accumulator
%     if ~isempty( ...                                        % if display_url exists
%             fake_news.statuses(ii).status.entities.urls) && ...
%             isfield(fake_news.statuses(ii).status.entities.urls,'display_url')
%         durl = fake_news.statuses(ii).status.entities.urls.display_url;
%         durl = regexp(durl,'^(.*?)\/','match','once');      % get its domain name
%         dispUrls(ii) = durl(1:end-1);                       % add to dipUrls
%         furl = fake_news.statuses(ii).status.entities.urls.expanded_url;
% %         furl = expandUrl(furl,'RemoveParams',1);            % expand links
% %         expUrls(ii) = expandUrl(furl,'RemoveParams',1);     % one more time
%     end
end

%% 
% Now we can create the document term matrix. We will also do the same
% thing for embedded links. 
dict = unique([tokens{:}]);                                 % unique words
domains = unique(dispUrls);                                 % unique domains
domains(domains == '') = [];                                % remove empty string
links = unique(expUrls);                                    % unique links
links(links == '') = [];                                    % remove empty string
DTM = zeros(tweetscnt,length(dict));              % Doc Term Matrix
DDM = zeros(tweetscnt,length(domains));           % Doc Domain Matrix
DLM = zeros(tweetscnt,length(links));             % Doc Link Matrix
for ii = 1:tweetscnt                              % loop over tokens
    [words,~,idx] = unique(tokens{ii});                     % get uniqe words
    wcounts = accumarray(idx, 1);                           % get word counts
    cols = ismember(dict, words);                           % find cols for words
    DTM(ii,cols) = wcounts;                                 % unpdate DTM with word counts
    cols = ismember(domains,dispUrls(ii));                  % find col for domain
    DDM(ii,cols) = 1;                                       % increment DMM
    expanded = expandUrl(expUrls(ii));                      % expand links
    expanded = expandUrl(expanded);                         % one more time
    cols = ismember(links,expanded);                        % find col for link
    DLM(ii,cols) = 1;                                       % increment DLM
end
% DTM(:,ismember(dict,{'#','@'})) = [];                       % remove # and @
% dict(ismember(dict,{'#','@'})) = [];  
CostFunction=@(DTM) Sphere(DTM);      % Cost Function
% CostFunction = DDM;

nVar = 10;          % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=-10;         % Decision Variables Lower Bound
VarMax= 10;         % Decision Variables Upper Bound

%% Cultural Algorithm Settings

MaxIt=1000;         % Maximum Number of Iterations

nPop=50;            % Population Size

pAccept=0.35;                   % Acceptance Ratio
nAccept=round(pAccept*nPop);    % Number of Accepted Individuals

alpha=0.3;

beta=0.5;

%% Initialization

% Initialize Culture
Culture.Situational.Cost=inf;
Culture.Normative.Min=inf(VarSize);
Culture.Normative.Max=-inf(VarSize);
Culture.Normative.L=inf(VarSize);
Culture.Normative.U=inf(VarSize);

% Empty Individual Structure
empty_individual.Position=[];
empty_individual.Cost=[];

% Initialize Population Array
pop=repmat(empty_individual,nPop,1);

% Generate Initial Solutions
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
end

% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);

% Adjust Culture using Selected Population
spop=pop(1:nAccept);
Culture=AdjustCulture(Culture,spop);

% Update Best Solution Ever Found
BestSol=Culture.Situational;

% Array to Hold Best Costs
BestCost=zeros(MaxIt,1);

%% multi object Cultural Algorithm Main Loop

for it=1:MaxIt
    
    % Influnce of Culture
    for i=1:nPop
        
        % % 1st Method (using only Normative component)
%         sigma=alpha*Culture.Normative.Size;
%         pop(i).Position=pop(i).Position+sigma.*randn(VarSize);
        
        % % 2nd Method (using only Situational component)
%         for j=1:nVar
%            sigma=0.1*(VarMax-VarMin);
%            dx=sigma*randn;
%            if pop(i).Position(j)<Culture.Situational.Position(j)
%                dx=abs(dx);
%            elseif pop(i).Position(j)>Culture.Situational.Position(j)
%                dx=-abs(dx);
%            end
%            pop(i).Position(j)=pop(i).Position(j)+dx;
%         end
        
        % % 3rd Method (using Normative and Situational components)
        for j=1:nVar
          sigma=alpha*Culture.Normative.Size(j);
          dx=sigma*randn;
          if pop(i).Position(j)<Culture.Situational.Position(j)
              dx=abs(dx);
          elseif pop(i).Position(j)>Culture.Situational.Position(j)
              dx=-abs(dx);
          end
          pop(i).Position(j)=pop(i).Position(j)+dx;
        end        
        
        % % 4th Method (using Size and Range of Normative component)
%         for j=1:nVar
%           sigma=alpha*Culture.Normative.Size(j);
%           dx=sigma*randn;
%           if pop(i).Position(j)<Culture.Normative.Min(j)
%               dx=abs(dx);
%           elseif pop(i).Position(j)>Culture.Normative.Max(j)
%               dx=-abs(dx);
%           else
%               dx=beta*dx;
%           end
%           pop(i).Position(j)=pop(i).Position(j)+dx;
%         end        
        
        pop(i).Cost=CostFunction(pop(i).Position);
        
    end
    
    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);

    % Adjust Culture using Selected Population
    spop=pop(1:nAccept);
    Culture=AdjustCulture(Culture,spop);

    % Update Best Solution Ever Found
    BestSol=Culture.Situational;
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

%% Results

figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
scores=T(:,3);  
scores=table2array(scores);