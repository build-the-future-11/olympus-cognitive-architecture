#setwd("/Users/uctpdwi/Dropbox/work/sync/Documents/projects/rankings AER PP/emp/journal IF")
setwd(dir_data_journalif)

#library(csranks)

# load data
load("IFdata.RData")
journalIF <- journalIF[!is.na(journalIF$IF) & journalIF$IF>0,]
journalIF$rank <- xrank(journalIF$IF)
journalIF <- journalIF[order(journalIF$rank),]

###################### compute CS

	# compute marginal CS for all journals
	CS_marg <- csranks(journalIF$IF, journalIF$SE, coverage=0.95, stepdown=TRUE, simul=FALSE, R=1000, seed=101)
	journalIF$margCSL <- CS_marg$L
	journalIF$margCSU <- CS_marg$U

	# compute simultaneous CS for all journals
	CS_simul <- csranks(journalIF$IF, journalIF$SE, coverage=0.95, stepdown=TRUE, simul=TRUE, R=1000, seed=101)
	journalIF$simulCSL <- CS_simul$L
	journalIF$simulCSU <- CS_simul$U

	# compute marginal CS for journals included in "top 30" of Kalaitzidakis et al. (2003)
	top30names <- c("JEL", "QJE", "JEP", "JFE", "JPE", "Econometrica", "AER", "REStud", "RESTAT", "JLE", "JEEM", "JHumRes", "J INT ECON", "EconJ", "JMonEcon", "JEconometrics", "RAND J ECON", "J BUS ECON STAT", "JPubEcon", "JAppliedEconmet", "EUR ECON REV", "IER", "JET", "OXFORD B ECON STAT", "GAME ECON BEHAV", "J ECON DYN CONTROL", "SCAND J ECON", "ECONOMET THEOR", "Econ Theory", "EL")
	journalIF.top30 <- journalIF[journalIF$journal %in% top30names,]
	journalIF.top30$rank <- xrank(journalIF.top30$IF)

	CS_marg.top30 <- csranks(journalIF.top30$IF, journalIF.top30$SE, coverage=0.95, stepdown=TRUE, simul=FALSE, R=1000, seed=101)
	journalIF.top30$margCSL <- CS_marg.top30$L
	journalIF.top30$margCSU <- CS_marg.top30$U

	# compute simul CS for top 30 journals
	CS_simul <- csranks(journalIF.top30$IF, journalIF.top30$SE, coverage=0.95, stepdown=TRUE, simul=TRUE, R=1000, seed=101)
	journalIF.top30$simulCSL <- CS_simul$L
	journalIF.top30$simulCSU <- CS_simul$U

	# CS for the 5-best of the top 30 journals
	CS_lower <- csranks(journalIF.top30$IF, journalIF.top30$SE, coverage=0.95, cstype = "lower", stepdown=TRUE, simul=TRUE, R=1000, seed=101)
	journalIF.top30$lowerCSL <- CS_lower$L
	journalIF.top30$lowerCSU <- CS_lower$U
	journalIF.top30$ind5best <- (CS_lower$L <= 5)

	# CS for the 25-worst of the top 30 journals
	CS_upper <- csranks(journalIF.top30$IF, journalIF.top30$SE, coverage=0.95, cstype = "upper", stepdown=TRUE, simul=TRUE, R=1000, seed=101)
	journalIF.top30$upperCSL <- CS_upper$L
	journalIF.top30$upperCSU <- CS_upper$U
	journalIF.top30$ind25worst <- (CS_upper$U > 5)

	save(journalIF, journalIF.top30, file="journalIFrankings.RData")



###################### plot CS

	#library(ggplot2)
	p <- nrow(journalIF)


	# ------- all journals

		# plot data
		plotdata <- ggplot(journalIF, aes(x=rank,y=IF)) + 
				geom_point(size=0.3) + 
				geom_errorbar(aes(ymin=IF-2*SE,ymax=IF+2*SE)) + 
				scale_x_continuous(name = "journal", limits = c(1, p), breaks = unique(c(1, seq(25, 200, by = 25), p)), labels = unique(c(1, seq(25, 200, by = 25), p))) +
				ylab(NULL) + theme_bw() + labs(title = "2011 Impact Factor", subtitle = "(with +/- 2*SE)")
		
		ggplot2::ggsave(paste0(dir_results, "journalIFdata.pdf"), plot=plotdata)

		# plot marginal CS 
		pl <- ggplot(data = journalIF, aes(x = reorder(journal, rank), y = rank)) +
				theme_bw() + 
				geom_point(size=0.02) + geom_errorbar(aes(ymin = margCSL, ymax = margCSU), position = position_dodge(0.1)) +
		        theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
		        coord_flip() +
		        scale_y_continuous(name = "rank", limits = c(1, p), breaks = unique(c(1, seq(25, 200, by = 25), p)), labels = unique(c(1, seq(25, 200, by = 25), p))) + 
		        xlab("journal") + labs(title = "Ranking of All Journals by 2011 Impact Factor", subtitle = "(with 95% marginal confidence sets)")
		
		ggplot2::ggsave(paste0(dir_results, "journalIFrankingall.pdf"), plot=pl)

		# plot simultaneous CS 
		pl <- ggplot(data = journalIF, aes(x = reorder(journal, rank), y = rank)) +
				theme_bw() + 
				geom_point(size=0.02) + geom_errorbar(aes(ymin = simulCSL, ymax = simulCSU), position = position_dodge(0.1)) +
		        theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
		        coord_flip() +
		        scale_y_continuous(name = "rank", limits = c(1, p), breaks = unique(c(1, seq(25, 200, by = 25), p)), labels = unique(c(1, seq(25, 200, by = 25), p))) + 
		        xlab("journal") + labs(title = "Ranking of All Journals by 2011 Impact Factor", subtitle = "(with 95% simultaneous confidence sets)")
		
		ggplot2::ggsave(paste0(dir_results, "journalIFrankingsimulall.pdf"), plot=pl)

		


	# ------- "top 30" journals

		# plot data
		plotdata.top30 <- ggplot(journalIF.top30, aes(x=reorder(journal, rank),y=IF)) + 
				geom_point() + 
				geom_errorbar(aes(ymin=IF-2*SE,ymax=IF+2*SE)) + 
				ylab(NULL) + theme_bw() + xlab(NULL) +
				scale_x_discrete(guide = guide_axis(angle = 60)) + labs(title = "2011 Impact Factor for 30 Journals", subtitle = "(with +/- 2*SE)")
		
		ggplot2::ggsave(paste0(dir_results, "journalIFdatatop30.pdf"), plot=plotdata.top30)

		# plot marg CS for top 30
		pl.top30 <- ggplot(data = journalIF.top30, aes(x = reorder(journal, rank), y = rank)) +
				theme_bw() + 
				geom_point() + geom_errorbar(aes(ymin = margCSL, ymax = margCSU), position = position_dodge(0.1)) +
		        coord_flip() +
		        scale_y_continuous(name = "rank", limits = c(1, 30), breaks = unique(c(1, seq(5, 30, by = 5))), labels = unique(c(1, seq(5, 30, by = 5)))) +
		        xlab(NULL) + labs(title = "Ranking of 30 Journals by 2011 Impact Factor", subtitle = "(with 95% marginal confidence sets)")

		ggplot2::ggsave(paste0(dir_results, "journalIFrankingtop30.pdf"), plot=pl.top30)

		# plot simul CS
		pl.top30 <- ggplot(data = journalIF.top30, aes(x = reorder(journal, rank), y = rank)) +
				theme_bw() + 
				geom_point() + geom_errorbar(aes(ymin = simulCSL, ymax = simulCSU), position = position_dodge(0.1)) +
		        coord_flip() +
		        scale_y_continuous(name = "rank", limits = c(1, 30), breaks = unique(c(1, seq(5, 30, by = 5))), labels = unique(c(1, seq(5, 30, by = 5)))) +
		        xlab(NULL) + labs(title = "Ranking of 30 Journals by 2011 Impact Factor", subtitle = "(with 95% simultaneous confidence sets)")
		ggplot2::ggsave(paste0(dir_results, "journalIFrankingsimultop30.pdf"), plot=pl.top30)

		# plot taubest CS
		pl.taubest <- ggplot(data = journalIF.top30, aes(x = reorder(journal, rank), y = rank, color=1*ind5best)) +
				theme_bw() + 
				geom_point() + geom_errorbar(aes(ymin = lowerCSL, ymax = lowerCSU), position = position_dodge(0.1)) +
		        coord_flip() +
		        scale_y_continuous(name = "rank", limits = c(1, 30), breaks = unique(c(1, 5, seq(5, 30, by = 5))), labels = unique(c(1, 5, seq(5, 30, by = 5)))) +
		        xlab(NULL) +
		    	geom_hline(yintercept = 5, size=0.5) +
		    	theme(legend.position="none") + labs(title = "Ranking of 30 Journals by 2011 Impact Factor", subtitle = "(with 95% simultaneous confidence sets)")
		ggplot2::ggsave(paste0(dir_results, "journalIFrankingtaubesttop30.pdf"), plot=pl.taubest)

		# plot tauworst CS
		pl.tauworst <- ggplot(data = journalIF.top30, aes(x = reorder(journal, rank), y = rank, color=1*ind25worst)) +
				theme_bw() + 
				geom_point() + geom_errorbar(aes(ymin = upperCSL, ymax = upperCSU), position = position_dodge(0.1)) +
		        coord_flip() +
		        scale_y_continuous(name = "rank", limits = c(1, 30), breaks = unique(c(1, 5, seq(5, 30, by = 5))), labels = unique(c(1, 5, seq(5, 30, by = 5)))) +
		        xlab(NULL) +
		    	geom_hline(yintercept = 6, size=0.5) +
		    	theme(legend.position="none") + labs(title = "Ranking of 30 Journals by 2011 Impact Factor", subtitle = "(with 95% simultaneous confidence sets)")
		ggplot2::ggsave(paste0(dir_results, "journalIFrankingtauworsttop30.pdf"), plot=pl.tauworst)

