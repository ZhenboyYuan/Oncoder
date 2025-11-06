# Preprocessing the normal plasma cfDNA methylation matrix 
options(timeout = 5000)
library(GEOquery)
geoMat_plasma <- getGEO("GSE40279")[[1]]

normal_plasma <- exprs(geoMat_plasma)
metadata_plasma <- pData(geoMat_plasma)
metadata_plasma <- metadata_plasma[, c("title", "geo_accession")]
metadata_plasma <- metadata_plasma[match(metadata_plasma$geo_accession, colnames(normal_plasma)),]

write.table(normal_plasma ,file = "normal_plasma.tsv",sep = "\t",quote = FALSE)
write.table(metadata_plasma,"metadata_normal_plasma.tsv",sep = "\t",quote = FALSE)


# Preprocessing the cfDNA methylation matrix from 31 prostate cancer samples.
library(data.table)
options(timeout = 5000)
ctProstate <- fread("/path/to/GSE108462_Unnormalised_signal.csv.gz")
ctProstate <- as.data.frame(ctProstate)
rownames(ctProstate) <- ctProstate$ID_REF
ctProstate <- ctProstate[,-1]

ctProstate_pvalue <- ctProstate[,grepl('Detection Pval', colnames(ctProstate))]
ctProstate_Unmethylated <- ctProstate[,grepl('Unmethylated Signal', colnames(ctProstate))]
ctProstate_Methylated <- ctProstate[,grepl('Methylated Signal', colnames(ctProstate))]

probes_to_keep <- which(! rowSums(ctProstate_pvalue > 0.01) >= 1)
ctProstate_Unmethylated <- ctProstate_Unmethylated[probes_to_keep,]
ctProstate_Methylated <- ctProstate_Methylated[probes_to_keep,]

ctProstate_norm <- ctProstate_Methylated / (ctProstate_Methylated + ctProstate_Unmethylated + 100)

# processing metadata_ctProstate
library(dplyr)
metadata_ctProstate <- pData(geoMat_ctProstate) %>%
    select(title, geo_accession, group = characteristics_ch1, tissue = characteristics_ch1.1, ID = characteristics_ch1.3, phenotype = characteristics_ch1.11, timepoint = characteristics_ch1.9,     batch = `cohort:ch1`) %>%   
    mutate(across(c(group, tissue, ID, phenotype, timepoint), ~ sub("^(diagnosis|tissue|individual id|phenotype|visit time point): ", "", .)), phenotype = sub(" to AA treatment", "", phenotype),     timepoint = as.numeric(timepoint))

colnames(ctProstate_norm) <- metadata_ctProstate$geo_accession

filter_and_sort <- function(data, pheno_type) {data[data$phenotype == pheno_type, ] %>% arrange(ID, timepoint)}
metadata_ctProstate_Resistant <- filter_and_sort(metadata_ctProstate, 'Resistant') 
metadata_ctProstate_Sensitive <- filter_and_sort(metadata_ctProstate, 'Sensitive')

write.table(ctProstate_norm[,metadata_ctProstate_Resistant$geo_accession],"ctProstate_Resistant.tsv",sep = "\t",quote = FALSE)
write.table(ctProstate_norm[,metadata_ctProstate_Sensitive$geo_accession],"ctProstate_Sensitive.tsv",sep = "\t",quote = FALSE)
write.table(metadata_ctProstate_Resistant,"metadata_ctProstate_Resistant.tsv",sep = "\t",quote = FALSE)
write.table(metadata_ctProstate_Sensitive,"metadata_ctProstate_Sensitive.tsv",sep = "\t",quote = FALSE)
