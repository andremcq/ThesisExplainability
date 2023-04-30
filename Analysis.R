# Script used to perform scan mix analysis
library(data.table)
library(ggplot2)
library(stringr)
library(lubridate)
library(ggthemes)
library(gridExtra)
library(zoo)
library(bit64)
library(dplyr)
library(ggrepel)
library(jpeg)
library(magick)

df = fread("DataSetMetaData.csv")

setDT(df)

df <- unique(df)

p1<-ggplot(data=df, aes(x=Type, y=Count, fill=Type)) +
  ggtitle("Scan Mix") +
  theme(plot.title = element_text(hjust = 0.5))+
  geom_bar(stat="identity")+
  xlab("Scan Classification")+
  ylab("Scan Count") +
  scale_fill_brewer(palette = "Pastel1")+
  theme(axis.text.x = element_text(angle = 45))

ggsave(plot = p1, filename = paste0("ScanMix.jpg"), width = 20, height = 15, units = "cm", dpi = 200)

# Set the working directory to the parent folder containing the subfolders
setwd("./Data/brain_tumor_all")

# Get a list of all jpeg files in the subfolders
files <- list.files(pattern = "\\.jpg$", recursive = TRUE, full.names = TRUE)

# Define a function to get the dimensions of an image
get_image_dimensions <- function(file) {
  #print(file)
  img <- readJPEG(file)
  return(paste0(dim(img)[1], "x", dim(img)[2]))
}

# Apply the function to each file in the list
dimensions <- lapply(files, get_image_dimensions)

# Combine the file names and dimensions into a data frame
result <- data.frame(unlist(files), unlist(dimensions))

names(result)[1] <- "files"
names(result)[2] <- "dimensions"
setDT(result)
result$files = NULL

counts <- result[, .N, by = dimensions]
counts <- counts[N < 11, dimensions := "Other"]

othercounts <- counts[dimensions == "Other"]
maincounts <- counts[dimensions != "Other"]

othercounts <- othercounts[, .N, by = dimensions]

allcounts <- rbind(othercounts, maincounts)

p2<-ggplot(data=allcounts, aes(x=dimensions, y=N, fill=dimensions), stat="identity", alpha=0.5) +
  ggtitle("Scan Dimension Mix") +
  scale_fill_brewer(palette="Accent") +
  theme(plot.title = element_text(hjust = 0.5))+
  geom_bar(stat="identity")+
  xlab("Scan Dimension")+
  ylab("Scan Dimension Count") +
  scale_fill_brewer(palette = "Pastel1") +
  theme(axis.text.x = element_text(angle = 45))

ggsave(plot = p2, filename = paste0("ScanDims.jpg"), width = 20, height = 15, units = "cm", dpi = 200)
