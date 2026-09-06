#install.packages(c("csranks", "gridExtra", "ggplot2"))
#library(csranks)
#library(gridExtra)
#library(ggplot2)    

# load data
topunis <- read.csv(paste0(dir_results, "top100unis.csv"), header=TRUE)
p <- nrow(topunis)

# compute confidence sets
topunis <- topunis[order(topunis$n, decreasing=TRUE),][1:100,]
topunis$IFrank <- xrank(topunis$IFmean)
# topunis$CTrank <- xrank(topunis$CTmean)

CS_marg <- csranks(topunis$IFmean, topunis$IFse, coverage=0.95, stepdown=TRUE, simul=FALSE, R=1000, seed=101)
topunis$IFmargL <- CS_marg$L
topunis$IFmargU <- CS_marg$U

CS_simul <- csranks(topunis$IFmean, topunis$IFse, coverage=0.95, stepdown=TRUE, simul=TRUE, R=1000, seed=101)
topunis$IFsimulL <- CS_simul$L
topunis$IFsimulU <- CS_simul$U

# CS_marg <- csranks(topunis$CTmean, topunis$CTse, coverage=0.95, stepdown=TRUE, simul=FALSE, R=1000, seed=101)
# topunis$CTmargL <- CS_marg$L
# topunis$CTmargU <- CS_marg$U

# plot rankings and marginal confidence sets for IF
plotIFall <- ggplot(topunis, aes(x=reorder(name,IFrank),y=IFrank)) + geom_point() + geom_errorbar(aes(ymin=IFmargL,ymax=IFmargU)) + labs(title = "Ranking universities by impact factor", subtitle = "(with 95% marginal confidence sets)") + xlab(NULL) + ylab(NULL) + theme_bw() + coord_flip() + scale_y_continuous(name = "rank", limits = c(1, p), breaks = unique(c(1, seq(5, 100, by = 5))), labels = unique(c(1, seq(5, 100, by = 5))))
ggplot2::ggsave(paste0(dir_results, "topunisIFrankingall.pdf"), plot=plotIFall, width=8, height=12)

# plot rankings and simultaneous confidence sets for IF
plotIFall <- ggplot(topunis, aes(x=reorder(name,IFrank),y=IFrank)) + geom_point() + geom_errorbar(aes(ymin=IFsimulL,ymax=IFsimulU)) + labs(title = "Ranking universities by impact factor", subtitle = "(with 95% simultaneous confidence sets)") + xlab(NULL) + ylab(NULL) + theme_bw() + coord_flip() + scale_y_continuous(name = "rank", limits = c(1, p), breaks = unique(c(1, seq(5, 100, by = 5))), labels = unique(c(1, seq(5, 100, by = 5))))
ggplot2::ggsave(paste0(dir_results, "topunisIFrankingsimulall.pdf"), plot=plotIFall, width=8, height=12)

# mu <- max(topunis$IFmargU[topunis$IFrank<=50])
# plotIF50.2 <- ggplot(subset(topunis, IFrank<=50), aes(x=reorder(name,IFrank),y=IFrank)) + geom_point() + geom_errorbar(aes(ymin=IFmargL,ymax=IFmargU)) + labs(title = "Ranking universities by impact factor (showing only top 50)", subtitle = "(with 95% marginal confidence sets)") + xlab("") + ylab(NULL) + theme_bw() + coord_flip() + scale_y_continuous(name = "rank", limits = c(1, mu), breaks = unique(c(1, seq(5, 50, by = 5))), labels = unique(c(1, seq(5, 50, by = 5))))
# ggplot2::ggsave("topunisIFranking.pdf", plot=plotIF50.2, width=8, height=7)

plotIF50.1 <- ggplot(topunis, aes(x=reorder(name,IFrank),y=IFmean)) + geom_point() + geom_errorbar(aes(ymin=IFmean-2*IFse,ymax=IFmean+2*IFse)) + labs(title = "Impact Factor", subtitle = "(with +/- 2*SE)") + xlab("") + ylab(NULL) + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggplot2::ggsave(paste0(dir_results, "topunisIFestim.pdf"), plot=plotIF50.1, width=12, height=8)

# plot rankings and confidence sets for CT
# plotCTall <- ggplot(topunis, aes(x=reorder(name,CTrank),y=CTrank)) + geom_point() + geom_errorbar(aes(ymin=CTmargL,ymax=CTmargU)) + labs(title = "Ranking universities by impact factor", subtitle = "(with 95% marginal confidence sets)") + xlab("university") + ylab(NULL) + theme_bw() + coord_flip() + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank()) + scale_y_continuous(name = "rank", limits = c(1, p), breaks = unique(c(1, seq(25, 100, by = 25))), labels = unique(c(1, seq(25, 100, by = 25))))
# ggplot2::ggsave("topunisCTranking.all.pdf", plot=plotCTall, width=8, height=6)

# mu <- max(topunis$CTmargU[topunis$CTrank<=25])
# plotCT50.2 <- ggplot(subset(topunis, CTrank<=25), aes(x=reorder(name,CTrank),y=CTrank)) + geom_point() + geom_errorbar(aes(ymin=CTmargL,ymax=CTmargU)) + labs(title = "Ranking universities by citation count (showing only top 25)", subtitle = "(with 95% marginal confidence sets)") + xlab("") + ylab(NULL) + theme_bw() + coord_flip() + scale_y_continuous(name = "rank", limits = c(1, mu), breaks = unique(c(1, seq(5, 25, by = 5))), labels = unique(c(1, seq(5, 25, by = 5))))
# ggplot2::ggsave("topunisCTranking.pdf", plot=plotCT50.2, width=8, height=6)

# plotCT50.1 <- ggplot(topunis, aes(x=reorder(name,CTrank),y=CTmean)) + geom_point() + geom_errorbar(aes(ymin=CTmean-2*CTse,ymax=CTmean+2*CTse)) + labs(title = "Citation Count", subtitle = "(with +/- 2*SE)") + xlab("") + ylab(NULL) + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# ggplot2::ggsave("topunisCTestim.pdf", plot=plotCT50.1, width=12, height=8)




