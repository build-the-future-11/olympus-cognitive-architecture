topunis <- read.csv(paste0(dir_results, "topunis.csv"), header=TRUE)

topunis$X <- NULL
topunis2 <- read.csv(paste0(dir_data_repec, "topuni_coded.csv"), header=TRUE)

total <- merge(topunis, topunis2, by="instname")
total <- subset(total, uni == 1)

topunis3 <- read.csv(paste0(dir_data_repec, "rename.csv"), header=TRUE)
total <- merge(total, topunis3, by="instname")

total <- total[, c(ncol(total), 1:(ncol(total) - 1))]
total$instname <- NULL
total$uni <- NULL
total <- total[order(total$n, decreasing=TRUE),]

write.csv(total, paste0(dir_results, "top100unis.csv"), row.names=FALSE)
