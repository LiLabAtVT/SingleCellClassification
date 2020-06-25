##This script will integrate merged data using the Seurat
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
merged_data_rds = args[1]
dim_num = args[2]
output_dir = args[3]
working_dir = args[4]

##step 0 extract feature number
merged_data_obj <- readRDS(merged_data_rds)
nfeature <- nrow(GetAssayData(object = merged_data_obj))
##step 1 find variable features
exp_data.list <- SplitObject(merged_data_obj,split.by = 'orig.ident')
exp_data.list <- lapply(X = exp_data.list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = nfeature) 
})
##step 2 conduct integration
exp_data.anchors <- FindIntegrationAnchors(object.list = exp_data.list, dims = 1:dim_num,anchor.features = nfeature)
exp_data.combined <- IntegrateData(anchorset = exp_data.anchors, dims = 1:dim_num)
DefaultAssay(exp_data.combined) <- "integrated"
saveRDS(exp_data.combined, paste(working_dir,'/merged_data_after_integration_obj.rds',sep = ''))
##get assay data
assay_data <- GetAssayData(object = exp_data.combined)
assay_data <- as.matrix(assay_data)
##updation 041420 ##generate matrix to save
saveRDS(assay_data, file = paste(working_dir,'/merged_data_after_integration_mtx.rds',sep = ''))
assay_data_mtx_format <- as.matrix(assay_data)
write.csv(assay_data_mtx_format,paste(output_dir,'/merged_data_after_integration_mtx.csv',sep = ''))







