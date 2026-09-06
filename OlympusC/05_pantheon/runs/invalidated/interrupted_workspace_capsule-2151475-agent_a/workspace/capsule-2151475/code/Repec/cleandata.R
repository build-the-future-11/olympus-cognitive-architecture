setwd(dir_data_repec)
file <- "originaldata/repec_raw.txt"
no_col <- max(count.fields(file, sep = " "))

# save original dataset in RData format
dat.raw <- data.frame(read.table(file, sep=" ", fill=TRUE, header = F, col.names=1:no_col))
colnames(dat.raw)[1] <- "affiliation"
colnames(dat.raw)[2] <- "author"
colnames(dat.raw)[3] <- "type"
save(dat.raw, file="repec.raw.RData")

# compute means and ses
n <- apply(dat.raw, 1, function(x) sum(!is.na(x)))-3
me <- rowMeans(dat.raw[,-(1:3)], na.rm=TRUE)
sde <- apply(dat.raw[,-(1:3)], 1, function(x) sd(x, na.rm=TRUE))
dat.raw$n <- n
dat.raw$mean <- me
dat.raw$sd <- sde
dat.raw$se <- dat.raw$sd / sqrt(dat.raw$n)

# remove authors with only one paper
dat.raw <- dat.raw[n>1,]


############### authors dataset

	# create aggregated dataset
	dat.raw.small <- dat.raw[,c("author", "affiliation", "type", "n", "mean", "se")]
	X1 <- dat.raw.small[dat.raw.small$type=="IF",]
	X2 <- dat.raw.small[dat.raw.small$type=="CT",]
	stopifnot(all(X2[,1]==X1[,1]) & all(X2[,2]==X1[,2]) & all(X2[,4]==X1[,4]))
	dat <- cbind(X1[,-3], X2[5:6])
	colnames(dat)[-(1:3)] <- c("IFmean", "IFse", "CTmean", "CTse")
	authors <- dat


############### institutions dataset

	institutions <- data.frame()

	# split up different affiliations, into a data frame
#	aff1 <- strsplit(dat.raw$affiliation, "*", fixed=TRUE)
#	naff <- unlist(lapply(aff1,length))
  aff1 = str_split(dat.raw$affiliation, fixed("*"), simplify = T)	
  
  # for each column of aff1, further split into affiliation and weight
  aff2 = str_split(aff1[,1], fixed("+"), simplify = T, n = 2)
  if (ncol(aff1) >= 2) {
    for (col in 2:ncol(aff1)) {
      aff2 = cbind(aff2, str_split(aff1[,col], fixed("+"), simplify = T, n = 2))
    }
  }
  
  # split affiliation further into type and code
  # cols with an odd number of ordering in aff2 are affiliation strings
  aff3 = str_split(aff2[,1], fixed(":"), simplify = T)
  if (ncol(aff2) >= 3) {
    for (col in seq(3, ncol(aff2) - 1, 2)) {
      aff3 = cbind(aff3, str_split(aff2[,col], fixed(":"), simplify = T))
    }
  }
  # each col should split into 3
  stopifnot("aff2 was not split properly" = ncol(aff3) == 3 * ncol(aff2) / 2)
  
  # from aff3:
  # - combine the first 2 in each group of 3 cols into institution type
  # - keep the 3rd col in each group of 3 cols as institution code
  # from aff2:
  # - add the weight back to the data frame
  aff4 = cbind(aff3[,3], paste0(aff3[,1], ":", aff3[,2]), aff2[,2])
  names_aff4 = c("instcode1", "insttype1", "weight1")
  if (ncol(aff3) > 3) {
    for (col in seq(4, ncol(aff3) - 2, 3)) {
      aff4 = cbind(aff4, cbind(aff3[,col+2], paste0(aff3[,col], ":", aff3[,col+1]), aff2[,(col+2)/3*2]))
      n_group = (col + 2) / 3
      names_aff4 = c(names_aff4, paste0(c("instcode", "insttype", "weight"), n_group))
    }
  }
  colnames(aff4) = names_aff4
  # replace empty institution type (now ":") with ""
  aff4[aff4 == ":"] = ""
  stopifnot("aff4 was not constructed properly" = ncol(aff4) == ncol(aff3))
  
  # reshape data so that each row corresponds to one affiliation 
  max_naff = ncol(aff4) / 3
  aff5 = data.table(cbind(aff4, dat.raw[,c("type","author")]))
  aff5 = melt(aff5, id.vars = c("type", "author"),
              measure.vars = list(paste0("instcode", 1:max_naff), paste0("insttype", 1:max_naff), paste0("weight", 1:max_naff)),
              value.name = c("instcode", "insttype", "weight"))
  aff5[, variable := NULL] # remove extra col from melt
  
  # merge with information from dat.raw
  institutions = merge.data.table(aff5, cbind(dat.raw[c("type", "author")], 
                                              dat.raw[grepl("X",names(dat.raw))]),
                                  by = c("type", "author"), all = T)

  institutions = as.data.frame(institutions)
	# remove authors without weights
  institutions$weight <- as.numeric(institutions$weight)
	indX <- grepl("X",names(institutions))
	indI <- complete.cases(institutions[,!indX])
	institutions <- institutions[indI,]

	# calculate sums
	institutions[,indX] <- lapply(institutions[,indX], as.numeric)
	institutions$n <- apply(institutions[,indX], 1, function(x) sum(!is.na(x)))
	institutions$wsum <- institutions$weight * rowSums(institutions[,indX], na.rm=TRUE)
	institutions$wsum2 <- institutions$weight^2 * rowSums(institutions[,indX]^2, na.rm=TRUE)

	# remove institutions that are not RePEc:edi
	indR <- institutions$insttype=="RePEc:edi"
	institutions <- institutions[indR,]

	# load institution names
	instnames <- read.csv("originaldata/inst.csv", header=FALSE)
	colnames(instnames) <- c("instcode", "instname")

	# merge institutions codes and names
	institutions.m <- merge(institutions, instnames, by="instcode")
	indX.m <- grepl("X",names(institutions.m))
	institutions.m <- institutions.m[,!indX.m]

	# group sums
	agginst <- aggregate(institutions.m[,c("n","wsum","wsum2")], by=list(instname=institutions.m$instname,type=institutions.m$type), sum)

	# group means and standard deviations
	agginst$mean <- agginst$wsum / agginst$n
	aggvar <- (agginst$wsum2 - agginst$wsum^2/agginst$n) / (agginst$n-1)
	agginst$se <- sqrt(aggvar / agginst$n)

	# remove banks and governments
	indnonuni <- grepl("Government ", agginst$instname, fixed = TRUE) | grepl("Bank ", agginst$instname, fixed = TRUE)
	unis <- agginst[!indnonuni,]

	# remove institutions with fewer than 100 articles
	unis <- unis[unis$n>100,]
	
	# reshape IF and CT from rows to columns
	Y1 <- unis[unis$type=="IF",]
	Y2 <- unis[unis$type=="CT",]
	stopifnot(all(Y1[,1]==Y2[,1]) & all(Y2[,3]==Y1[,3]))
	unis <- Y1[,-c(2,4,5)]
	unis$CTmean <- Y2$mean
	unis$CTse <- Y2$se
	colnames(unis)[3:4] <- c("IFmean", "IFse")
	rownames(unis) <- NULL

############### save datasets

	save(authors, institutions, unis, file="repec.RData")

	# save 100 most productive universities
	unis.top <- unis[order(unis$n,decreasing=TRUE),]
	write.csv(unis.top, paste0(dir_results, "topunis.csv"))


