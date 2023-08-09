library(clusterProfiler)
library(fgsea)
library(GOstats)
library(msigdb)
library(org.Mm.eg.db)
options(warn = -1)


# Input two columns of gene symbols, first letter capital, others lower case
impor_gene <- read.csv('important_genes.csv')[, 1]
impor_all <- read.csv('all_genes.csv')[, 1]
keytypes(org.Mm.eg.db)
gene_symbol <- impor_gene
gene_symbol_all <- impor_all
gene_ids_impor <- AnnotationDbi::select(org.Mm.eg.db, keys = as.character(gene_symbol), column = "ENTREZID", keytype = "SYMBOL")
gene_ids_all <- AnnotationDbi::select(org.Mm.eg.db, keys = as.character(gene_symbol_all), column = "ENTREZID", keytype = "SYMBOL")
sel <- gene_ids_impor
uni <- gene_ids_all
params <- new("GOHyperGParams", geneIds = sel, universeGeneIds = uni, annotation = "org.Mm.eg.db", ontology = "BP", pvalueCutoff = 0.01, conditional = F, testDirection = "over")
over <- hyperGTest(params)
b <- summary(over)
write.csv(b, "GOstats.csv")


total <- list()
impo <- data.frame(gene_ids_impor[, 1])
colnames(impo) <- c("gene_important")
for (i in 1:dim(b)[1]){
  a <- matrix(b[i, ])
  colnames(a) <- "GO"
  gos <- b$GOBPID[i]
  z <- mapIds(org.Mm.eg.db, gos, "SYMBOL", "GOALL", multiVals = "list")
  z <- data.frame(z)
  colnames(z) <- c("gene_important")
  geneName <- data.frame(intersect(x = z$gene_important, y = impo$gene_important))
  colnames(geneName) <- "GO"
  x <- rbind(a, geneName)
  total[i] <- list(x)
}
total2 <- list()
for (i in 1:length(total)){
  total2[[i]] <- as.matrix(total[[i]])
}
df2 <- do.call(cbind, lapply(lapply(total2, unlist), 'length<-', max(lengths(total2))))
write.csv(df2, "GOTest_total.csv")


df1 <- df2
ind <- c()
for (z in 2:dim(df1)[2]){
  if (as.integer(df1[6, z]) <= 200){
    if (!(z %in% ind)){
      ind <- c(ind, z)
    }
  }
}
df <- df1[, ind]
index <- c()
df_gene <- df[8:dim(df)[1], ]
for (i in 2:(dim(df)[2] - 1)){
  gene_li <- na.omit(df_gene[, i])
  for (j in (i+1):dim(df)[2]){
    gene_li2 <- na.omit(df_gene[, j])
    inter <- intersect(gene_li, gene_li2)
    big <- max(length(gene_li), length(gene_li2))
    if (length(gene_li) == big){
      if (length(inter)/length(gene_li) >= 0.75){
        if (!(i %in% index)){
          index <- c(index, i)
        }
        else if (length(inter)/length(gene_li2) >= 0.9){
          if (!(j %in% index)){
            index <- c(index, j)
          }
        }
      }
    }
    else{
      if (length(inter)/length(gene_li2) >= 0.75){
        if (!(j %in% index)){
          index <- c(index, j)
        }
        else if (length(inter)/length(gene_li) >= 0.9){
          if (!(j %in% index)){
            index <- c(index, j)
          }
        }
      }
    }
  }
}
df_reduce <- df[, -index]
write.csv(df_reduce, "GOTest_total_select.csv") # final output