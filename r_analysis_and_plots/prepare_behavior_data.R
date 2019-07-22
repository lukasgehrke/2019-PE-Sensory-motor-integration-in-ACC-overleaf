# prepare data IMT1 full paper submissions

# event level resolution
# load data from mocap event structure
fpath <- 'P:\\Lukas_Gehrke\\studies\\Visual_Maze_1\\data\\5_study_level\\visMaze_110comps\\design_matrices\\'
subjects <- c(1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,28,29,30,31,32)

df_all <- data.frame()
for (subject in subjects) {
  df <- as.data.frame(read.csv(paste(fpath, 's', toString(subject), '_design_matrix.csv', sep = '')))
  df$participant <- rep(subject, dim(df)[1])
  df_all <- rbind(df_all, df)
}
df <- df_all
df$participant <- as.factor(df$participant)
df$trial_run <- as.factor(df$trial_run)
df$maze <- as.factor(df$maze)
df$wall_change <- as.factor(df$wall_change)

df_agg <- aggregate(. ~ participant + trial_run + maze, df, mean)
df_agg <- df_agg[order(df_agg$participant),]

# subject level resolution
# much more interesting: merge df_agg with data mean_ratings, ptsot, sod
# does sketchmap predict inter_touch_time and distance covered and touch_duration?
# argumentation: robust mental representation leads to a confident maze exploration behavior
fpath <- 'P:\\Lukas_Gehrke\\studies\\Visual_Maze_1\\data_processing\\GLM\\reports\\'
load(paste(fpath, 'vm_all.RData', sep = ''))
# measures_of_int <- vm[,c(1,2,3,4,5,6,17,18,19,20,21)]
measures_of_int <- vm
measures_of_int <- measures_of_int[measures_of_int$Participant %in% subjects,]

# data cleaning
# find mismatches (somewhere 1 subject is wrong, s25 at row 277)
# subject 25 has 1 duplicate for some reason, delete it
measures_of_int <- measures_of_int[-277,]

df_all <- as.data.frame(cbind(df_agg, measures_of_int))
df <- df_all[,-c(1:3)]

## start cleaning copied from SC2018
# data cleaning similar to study SC2018 and throw out the same stuff and cite
# throw out trials with more than 10 head collisions
# --> 6 Trials --> do after all data is merged
df <- df[!df$Head_Collisions >= 10,]

# throw out trials with technical errors
df <- df[!(df$Participant == 22 & df$Run == 3 & df$Maze == 'U'),]
df <- df[!(df$Participant == 22 & df$Run == 3 & df$Maze == 'L'),]
df <- df[!(df$Participant == 31 & df$Run == 2 & df$Maze == 'U'),]
df <- df[!(df$Participant == 32 & df$Run == 2 & df$Maze == 'U'),]

# throw out all runs where the maze was not fully explored, progress != 1.0
df <- df[!df$progress < 1.0,]
## end cleaning copied from SC2018

# # remove outliers in df_all
# outliers_d <- boxplot(df_all$Duration)$out
# df_all <- df_all[-which(df_all$Duration %in% outliers_d),]
# 
# outliers_t <- boxplot(df_all$Hand_Touches)$out
# df_all <- df_all[-which(df_all$Hand_Touches %in% outliers_t),]
# 
# outliers_v <- boxplot(df_all$Velocity)$out
# df_all <- df_all[-which(df_all$Velocity %in% outliers_v),]
# 
# outliers_i <- boxplot(df_all$inter_touch_time)$out
# df_all <- df_all[-which(df_all$inter_touch_time %in% outliers_i),]

# save data
fpath_out <- 'P:\\Lukas_Gehrke\\studies\\Visual_Maze_1\\data_processing\\behavior\\'
fname_out <- 'behavior_IMT1.Rdata'
save(df, file = paste(fpath_out, fname_out))

