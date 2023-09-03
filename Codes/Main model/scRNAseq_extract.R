# Atlas
library(scRNAseq)
data <- HeOrganAtlasData(tissue = "Blood", ensembl = FALSE, location = TRUE)
X <- data@assays@data@listData$counts
y <- data@colData@listData$Cell_type_in_each_tissue
X_matrix <- as.matrix(X)

write.csv(X_matrix, "atlas_data.csv", col.names = F)
write.csv(y, "atlas_label.csv", col.names = F)
