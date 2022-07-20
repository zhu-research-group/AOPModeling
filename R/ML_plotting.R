
library(ggplot2)
library(dplyr)
library(tidyverse)
library(patchwork)
library(scales)
library(glue)

endpoint <- "H_HT_class"

hybrid_preds = read.csv(glue("../data/text/ML/{endpoint}_LOG:False_BALANCE:True_SPLIT:LOO/hybrid_preds.csv"))
aop_preds = read.csv(glue("../data/text/ML/{endpoint}_LOG:False_BALANCE:True_SPLIT:LOO/aop_preds.csv"))

tox.colors <- c("1"="red", "0"="green")
preds = hybrid_preds

hybrid_p1 <- preds %>%
  mutate(Hepatotoxicity = as.factor(Hepatotoxicity)) %>%
  ggplot(aes(color = Hepatotoxicity)) +
  geom_density(aes(x = Probability, fill=Hepatotoxicity), alpha=0.4, show.legend = FALSE) +
  geom_rug(aes(x = Probability), show.legend = FALSE) + scale_color_manual(values=tox.colors) + scale_fill_manual(values=tox.colors) + theme_linedraw()  + xlim(0, 1) + scale_x_continuous(breaks=c(0, 0.5, 1)) + labs(y = "Density")
 
hybrid_p2<-preds %>%
  arrange(desc(Probability)) %>%
  mutate(Hepatotoxicity = as.factor(Hepatotoxicity)) %>%
  mutate(Rank = 1:nrow(preds)) %>%
  ggplot(aes(color=Hepatotoxicity)) + 
  geom_rug(aes(y = Probability, color=Hepatotoxicity), show.legend = FALSE) +
  geom_point(aes(x = Rank, y = Probability)) + scale_color_manual(values=tox.colors) + scale_fill_manual(values=tox.colors) + theme_linedraw()


preds = aop_preds

aop_p1 <- preds %>%
  mutate(Hepatotoxicity = as.factor(Hepatotoxicity)) %>%
  ggplot(aes(color = Hepatotoxicity)) +
  geom_density(aes(x = Probability, fill=Hepatotoxicity), alpha=0.4, show.legend = FALSE) +
  geom_rug(aes(x = Probability), show.legend = FALSE) + scale_color_manual(values=tox.colors) + scale_fill_manual(values=tox.colors) + theme_linedraw()  + xlim(0, 1) + scale_x_continuous(breaks=c(0, 0.5, 1)) + labs(y = "Density")

aop_p2 <-preds %>%
  arrange(desc(Probability)) %>%
  mutate(Hepatotoxicity = as.factor(Hepatotoxicity)) %>%
  mutate(Rank = 1:nrow(preds)) %>%
  ggplot(aes(color=Hepatotoxicity)) + 
  geom_rug(aes(y = Probability, color=Hepatotoxicity), show.legend = FALSE) +
  geom_point(aes(x = Rank, y = Probability), show.legend = FALSE) + scale_color_manual(values=tox.colors) + scale_fill_manual(values=tox.colors) + theme_linedraw()

#patchwork <- (aop_p1 | aop_p2) / (hybrid_p1 | hybrid_p2)
patchwork <- (aop_p1 | hybrid_p1) / (aop_p2 | hybrid_p2)

patchwork
