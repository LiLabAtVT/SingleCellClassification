##This script will filer features using the Seurat
library(Seurat)

##set parsing argument
args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} #else if (length(args)==1) {
# default output file
#args[2] = "out.txt"
#}

##arguments
merged_or_integrated_data_rds = args[1]
fl_feat_proportion = args[2]
output_dir = args[3]

########################
##find the most variable features
ipt_object <- readRDS(merged_or_integrated_data_rds)
nfeat <- as.integer(nrow(GetAssayData(object = ipt_object))*as.numeric(fl_feat_proportion))
pbmc <- FindVariableFeatures(ipt_object, selection.method = "vst", nfeatures = nfeat)
##extract var feats
var_feats <- head(VariableFeatures(pbmc), nfeat)
##write to csv
write.csv(var_feats,paste(output_dir,'/var_',nfeat,'_feats.csv',sep = ''))








