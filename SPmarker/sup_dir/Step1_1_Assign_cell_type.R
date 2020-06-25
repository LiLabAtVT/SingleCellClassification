##This script will assign cell type to cells using the ICI index method
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
spec_score_file = args[2]
output_dir = args[3]
working_dir = args[4]


##Step 0 read spec and merged data
merged_data_obj <- readRDS(merged_data_rds)
ath_root_marker_spec_score <- read.csv(spec_score_file, row.names=1)
spec_scores <- ath_root_marker_spec_score

##Step 1 calculate the ici
ExpMat <- GetAssayData(merged_data_obj)
#get only marker gene expression
markergenes <- rownames(spec_scores)
markergenesk <- markergenes[markergenes%in%rownames(ExpMat)]
MarkerMat <- ExpMat[markergenesk,]
#get SPEC Identity matrix, and Nt vector (number of marker gene per tissue)
speck <- spec_scores[rownames(MarkerMat),]
specI <- speck;specI[specI>0]=1;specI[specI<1]=0
Nt <- apply(specI,2,sum)
#get marker identity matrix, (m by C)
markerI <- MarkerMat; markerI[markerI>0]=1; markerI[markerI<1]=0

# Actual ICI calculation
# A = sum of (gene expression values * spec score for gene,tissue)
# B = sum of markers present (sum gene over tissue)
# ICI = (A * B) / Nt**2

# multiply the transpose of the marker identities by the spec identities
B <- t(as.matrix(markerI))%*%as.matrix(specI)
# Cell by tissue matrix
A <- t(as.matrix(MarkerMat))%*%as.matrix(speck)
# ICI before normalization (top part of equation commented above)
numerator <- A * B
# Divide (normalize) ICI by the total number of markers for each tissue
ICI <- t(t(numerator)/(Nt**2))

#filter the ICI's by significance 
#sum the ICI scores ove the rows
rowsumk <- apply(ICI,1,sum)

#keep only the rows with ICI > 0
rowsumk <- rowsumk[rowsumk>0]
ICIn <- ICI[names(rowsumk),]

#normalized ICI for cells with ICI numbers.
ICIn <- ICIn/rowsumk
#check ICI norowsum
ICInrowsum <- apply(ICIn,1,sum)
#get ICI assigned cell types, ICI > 0.5 assign 1. otherwise assign 0.
ICIcall <- ICIn; ICIcall[ICIcall>0.5]=1; ICIcall[ICIcall<1]=0

##Step 2 write out results
write.csv(ICIcall,paste(working_dir,'/ICIcall.csv',sep = ''))
write.csv(ICIn,paste(output_dir,'/ICIn.csv',sep = ''))











