rm(list=ls())

#== set environment to run the code in
# local: run it on the local computer
# CodeOcean: for running on CodeOcean
envir = "CodeOcean"

#=== directories
loc_dir_main = getwd()
if (envir == "local") {
  dir_main = loc_dir_main
}

if (envir == "CodeOcean") {
  dir_main = ""
}

# data
dir_data_main = paste0(dir_main, "/data")
dir_data_repec = paste0(dir_data_main, "/Repec/")
dir_data_journalif = paste0(dir_data_main, "/journal IF/")

# code
dir_code_main = paste0(dir_main, "/code")
dir_code_repec = paste0(dir_code_main, "/Repec/")
dir_code_journalif = paste0(dir_code_main, "/journal IF/")

# outputs
dir_results = paste0(dir_main, "/results/")

#=== R packages
# if there is any issue installing the csranks package
# it can be installed by calling install_github("danielwilhelm/R-CS-ranks")
# as long as the devtools package is loaded
# if there is still error loading the csranks package, may be due to outdated versions of other packages called in csranks
# update those packages according to error message
list_pkgs = c("devtools", "csranks", "gridExtra", "ggplot2", 
              "readxl", "stringr", "data.table")
lapply(list_pkgs, library, character.only = T)

t_start = Sys.time()

#=== run journal IF code
source(paste0(dir_code_journalif, "convertdata.R"))
source(paste0(dir_code_journalif, "journalranking.R"))

#=== run Repec codes
source(paste0(dir_code_repec, "cleandata.R"))
source(paste0(dir_code_repec, "keeptop100.R"))
source(paste0(dir_code_repec, "emp.topunis.R"))

t_end = Sys.time()

# calculate run time
t_diff = t_end - t_start
t_diff


