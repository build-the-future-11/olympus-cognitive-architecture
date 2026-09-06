#library(readxl)
#setwd("/Users/uctpdwi/Dropbox/work/sync/Documents/projects/rankings AER PP/emp/trunk/journal IF")
setwd(dir_data_journalif)
filename <- "JEL_Data.xlsx"
sheetnames <- excel_sheets(filename)
mylist <- lapply(excel_sheets(filename), read_excel, path = filename)
names(mylist) <- sheetnames

substrRight <- function(x, n) {
	res <- NULL
	for (i in 1:length(x)) res <- c(res,substr(x[i], nchar(x[i])-n+1, nchar(x[i])))
	return(res)
}

journalIF <- NULL
for (s in sheetnames[-1]) {

	X <- as.data.frame(mylist[s])	
	last4rows <- as.matrix(X[((nrow(X)-3):nrow(X)),])
	stopifnot(sum(is.na(last4rows)) > 0.1*length(last4rows))
	X <- X[-((nrow(X)-3):nrow(X)),]

	lastcharscn <- substrRight(names(X), 5)
	stopifnot(sum(lastcharscn==".2011")==1)
	col2011 <- names(X)[lastcharscn==".2011"]

	journalIF <- rbind(journalIF, c(mean(X[,col2011]),sd(X[,col2011])/sqrt(nrow(X)),sd(X[,col2011]),nrow(X)))
}

colnames(journalIF) <- c("IF", "SE", "sigma", "n")

journalIF <- as.data.frame(journalIF)
journalIF$journal <- sheetnames[-1]
save(journalIF, file = paste0(dir_data_journalif, "IFdata.RData"))