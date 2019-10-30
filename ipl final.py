# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:40:21 2019

@author: Pranay Gaikwad
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, svm, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

Match = pd.read_csv('file:///C:/Users/sif-/Desktop/matches.csv')
Match.columns
Match = Match[Match.winner.notnull()]

#1 To find total number of match played
Total_match = []
    Team_count = pd.DataFrame()
Team = pd.DataFrame(Match["team1"].unique())
Team.columns = ["Team"]
for i in Team["Team"]:
    Team_count["T1"] = Match["team1"] == i
    Team_count["T2"] = Match["team2"] == i
    Team_count["T3"] = True
    for j in range(0,len(Team_count["T1"])):
        Team_count.iloc[j,2] = Team_count.iloc[j,0] or Team_count.iloc[j,1]
    Total = np.sum(Team_count["T3"])
    Total_match.append(Total)
Team["Total_match"] = Total_match

#2 To find proportion of match won
Matches_won = pd.pivot_table(Match, index = "winner", values = "id", aggfunc = len)
Matches_won["Team"] = Matches_won.index
Team = pd.merge(Team, Matches_won, left_on = "Team", right_on = "Team")
Team.columns = ['Team', 'Total_match', 'Total_win']
Team["Total_win_per"] = (Team["Total_win"]/Team["Total_match"])*100

#3 win when bat first
Team_bat_first = pd.pivot_table(Match, index = "team1", values = "id", aggfunc = len)
Team_bat_first.columns = ['Count_bat_first']
Team_bat_first["Team"] = Team_bat_first.index
Team = pd.merge(Team, Team_bat_first, on = 'Team')

Match["Bat_win_flag"] = Match["team1"] == Match["winner"]
Matches_won_bat = pd.pivot_table(Match, index = "team1", values = "Bat_win_flag", aggfunc = sum)
Matches_won_bat["Team"] = Matches_won_bat.index 
Team = pd.merge(Team, Matches_won_bat, on = "Team")
Team.columns = ['Team', 'Total_match', 'Total_win', 'Total_win_per', 'Count_bat_first', 'bat_win_1st']
Team["bat_win_1st_per"] = (Team["bat_win_1st"]/Team["Count_bat_first"])*100
#Team["Win_field_1st_per"] = (Team["Win_field_1st"]/Team["Total_win"])*100

#4 win when toss is won
toss = Match[['toss_winner', 'winner']]
toss["check"] = toss['toss_winner'] == toss['winner'] 
Matches_won_toss = pd.pivot_table(toss, index = "winner", values = "check", aggfunc = sum)
Matches_won_toss.columns = ["Match_won_toss_won"]
Matches_won_toss["Team"] = Matches_won_toss.index

Toss_won = pd.pivot_table(Match, index = 'toss_winner', values = 'id', aggfunc = len)
Toss_won.columns = ['Toss_won_count']
Toss_won["Team"] = Toss_won.index
Team = pd.merge(Team, Matches_won_toss, on = "Team")
Team = pd.merge(Team, Toss_won, on = "Team")
Team["Match_won_toss_won_per"] = (Team["Match_won_toss_won"]/Team["Toss_won_count"])*100

#5.1 Win per with all team
Match["Losser"] = np.where(Match["winner"] == Match["team1"], Match["team2"], Match["team1"])
Win_against_other_team = pd.pivot_table(Match, index = ["winner", "Losser"], values = "id", aggfunc = len) 
Win_against_other_team.reset_index(inplace=True)
Win_against_other_team.columns = ['winner', 'Losser', 'Win_against_loser']

#5.2 Loss per with all team
Loss_against_other_team = pd.pivot_table(Match, index = ["Losser", "winner"], values = "id", aggfunc = len) 
Loss_against_other_team.reset_index(inplace=True)
Loss_against_other_team.columns = [ 'Losser', 'winner', 'Lost_against_winner']

#5.3 loss win in 1 table
Loss_win_data = pd.merge(Win_against_other_team, Loss_against_other_team, left_on = ["winner", "Losser"], right_on = ["Losser", "winner"], how = "outer")
Loss_win_data['winner_x'] = np.where(Loss_win_data['winner_x'].isnull(), Loss_win_data['Losser_y'], Loss_win_data['winner_x'])
Loss_win_data['Losser_x'] = np.where(Loss_win_data['Losser_x'].isnull(), Loss_win_data['winner_y'], Loss_win_data['Losser_x'])
Loss_win_data.columns
Loss_win_data = Loss_win_data.drop('Losser_y',1)
Loss_win_data = Loss_win_data.drop('winner_y',1)
Loss_win_data.columns = ['winner', 'Losser', 'Win_against_loser', 'Lost_against_winner']

#5.4 Total matches
Loss_win_data['Win_against_loser'] = np.where(Loss_win_data['Win_against_loser'].isnull(), 0 ,Loss_win_data['Win_against_loser'])
Loss_win_data['Lost_against_winner'] = np.where(Loss_win_data['Lost_against_winner'].isnull(), 0, Loss_win_data['Lost_against_winner'])
Loss_win_data['Total_each_team'] = Loss_win_data['Win_against_loser'] + Loss_win_data['Lost_against_winner']
len(Loss_win_data['winner'].unique())
Loss_win_data['Win_against_loser_per'] = (Loss_win_data['Win_against_loser']/Loss_win_data['Total_each_team'])*100
Loss_win_data['Lost_against_winner_per'] = (Loss_win_data['Lost_against_winner']/Loss_win_data['Total_each_team'])*100

##Wrong mapping
##Wrong Mapping Win % and loss % to match data
#Match_new = pd.merge(Match, Loss_win_data[['winner', 'Losser', 'Win_against_loser_per', 'Lost_against_winner_per']], left_on = ['winner', 'Losser'], right_on = ['winner', 'Losser'], how = 'left')

#6.1Preparing Data for team1
Team1_data =  Match[['id', 'team1', 'toss_winner', 'team2', 'winner']]
Team1_data['Batting_flag'] = 1
Team1_data['toss_flag'] = np.where(Team1_data['team1'] == Team1_data['toss_winner'], 1, 0)
Team1_data['winner_flag'] = np.where(Team1_data['team1'] == Team1_data['winner'], 1, 0)

Team1_data = pd.merge(Team1_data, Loss_win_data[['winner', 'Losser', 'Win_against_loser_per']], left_on = ['team1', 'team2'], right_on = ['winner', 'Losser'], how = "left")
Team1_data = Team1_data.drop(['winner_y', 'Losser'], 1) 

Team1_data = pd.merge(Team1_data, Team[['Team', 'Total_win_per', 'bat_win_1st_per', 'Match_won_toss_won_per']], left_on = 'team1', right_on = 'Team', how = 'left')
Team1_data = Team1_data.drop("Team",1)
Team1_data.columns = ['id', 'team', 'toss_winner', 'Opponent_team', 'winner', 'Batting_flag',
                      'toss_flag', 'winner_flag', 'Win_against_opponent_per', 'Total_win_per',
                      'bat_win_1st_per', 'Match_won_toss_won_per']

#6.2 Preparing Data for team2
Team2_data =  Match[['id', 'team2', 'toss_winner', 'team1', 'winner']]
Team2_data['Batting_flag'] = 0
Team2_data['toss_flag'] = np.where(Team2_data['team2'] == Team2_data['toss_winner'], 1, 0)
Team2_data['winner_flag'] = np.where(Team2_data['team2'] == Team2_data['winner'], 1, 0)

Team2_data = pd.merge(Team2_data, Loss_win_data[['winner', 'Losser', 'Win_against_loser_per']], left_on = ['team2', 'team1'], right_on = ['winner', 'Losser'], how = "left")
Team2_data = Team2_data.drop(['winner_y', 'Losser'], 1) 

Team2_data = pd.merge(Team2_data, Team[['Team', 'Total_win_per', 'bat_win_1st_per', 'Match_won_toss_won_per']], left_on = 'team2', right_on = 'Team', how = 'left')
Team2_data = Team2_data.drop("Team",1)
Team2_data.columns = ['id', 'team', 'toss_winner', 'Opponent_team', 'winner', 'Batting_flag',
                      'toss_flag', 'winner_flag', 'Win_against_opponent_per', 'Total_win_per',
                      'bat_win_1st_per', 'Match_won_toss_won_per']


#6.3 combining the data
Model_data = pd.concat([Team1_data, Team2_data], axis = 0, sort = False)
Model_data = Model_data.drop('winner',1)
Model_data.isnull().sum()
Model_data.reset_index(inplace = True)
#Model_data.to_csv('C:/Users/Priyanka Gaikwad/Desktop/Analytics_club_project/ipl_2019/ipldata/Model_data.csv", index = False')


#Loss_win_data.to_csv("C:/Users/Priyanka Gaikwad/Desktop/Analytics_club_project/ipl_2019/ipldata/Loss_win_data.csv", index = False)
#Match_new.to_csv("C:/Users/Priyanka Gaikwad/Desktop/Analytics_club_project/ipl_2019/ipldata/Match_new.csv", index = False)
#Team.to_csv("C:/Users/Priyanka Gaikwad/Desktop/Analytics_club_project/ipl_2019/ipldata/Team.csv", index = False)


############
#Modelling
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, svm, neighbors#, cross_validation
#from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
##
Model_data = Model_data.drop('id', 1)
Model_data = Model_data.drop('toss_winner', 1)

#7 Label encoding
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()   
Model_data['team1']= le.fit_transform(Model_data['team']) 
Model_data['Opponent_team1']= le.transform(Model_data['Opponent_team'])

Team_mapping = Model_data[['team', 'team1']].drop_duplicates()
Team_mapping['opp_team'] = 'opp_' + Team_mapping['team']
Team_mapping = Team_mapping.sort_values(by = ['team1'], ascending = True)
Team_mapping.reset_index(inplace = True)

#8 One hot encoding
from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder()   
Team_1_hot = onehotencoder.fit_transform(Model_data[['team', 'Opponent_team']]).toarray()
Team_1_hot = pd.DataFrame(Team_1_hot)
##Giving Column names to one hot encoding
Col_names = Team_mapping['team']
Col_names = Col_names.append(Team_mapping['opp_team'])
Team_1_hot.columns = Col_names

Model_data = pd.concat([Model_data, Team_1_hot], axis = 1)
Model_data.columns
Model_data = Model_data.drop(['index', 'team', 'Opponent_team', 'team1', 'Opponent_team1'], 1)
X = Model_data.drop('winner_flag', 1)
Y = Model_data[['winner_flag']]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.20)

#9.1SVM
from sklearn.metrics import accuracy_score
lin_svm = svm.LinearSVC()
lin_svm.fit(X_train, y_train)
X_test['SVM_pred'] = lin_svm.predict(X_test)
print(accuracy_score(y_test,X_test['SVM_pred']))

#9.2Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
X_test['gnb_pred'] = gnb.predict(X_test.drop(['SVM_pred'],1))
print(accuracy_score(y_test,X_test['gnb_pred']))

#9.3Random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
X_test['rf_pred'] = rf.predict(X_test.drop(['SVM_pred', 'gnb_pred'],1))
print(accuracy_score(y_test,X_test['rf_pred']))

#9.4ensemble
X_test_mode = X_test[['SVM_pred', 'gnb_pred', 'rf_pred']].mode(axis = 1)
X_test = pd.concat([X_test, X_test_mode],1)
X_test.columns = ['Batting_flag','toss_flag',
'Win_against_opponent_per','Total_win_per',
'bat_win_1st_per','Match_won_toss_won_per',
'ChennaiSuperKings','DeccanChargers',
'DelhiCapitals','DelhiDaredevils',
'GujaratLions','KingsXIPunjab',
'KochiTuskersKerala','KolkataKnightRiders',
'MumbaiIndians','PuneWarriors',
'RajasthanRoyals','RisingPuneSupergiant',
'RoyalChallengersBangalore','SunrisersHyderabad',
'opp_ChennaiSuperKings','opp_DeccanChargers',
'opp_DelhiCapitals','opp_DelhiDaredevils',
'opp_GujaratLions','opp_KingsXIPunjab',
'opp_KochiTuskersKerala','opp_KolkataKnightRiders',
'opp_MumbaiIndians','opp_PuneWarriors',
'opp_RajasthanRoyals','opp_RisingPuneSupergiant',
'opp_RoyalChallengersBangalore','opp_SunrisersHyderabad',
'SVM_pred','gnb_pred',
'rf_pred','ensemble']
print(accuracy_score(y_test,X_test['ensemble']))
