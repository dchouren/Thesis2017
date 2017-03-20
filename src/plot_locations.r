
library(ggplot2)
geo_locs <- read.table('../2014_geo', header=FALSE, sep=',')
m <- qplot(xlab="Longitude",ylab="Latitude",main="2014 Photos",geom="blank",x=geo_locs$V2,y=geo_locs$V1,data=geo_locs,xlim=c(-74.052544,-73.740685), ylim=c(40.525070,40.889249))  + stat_bin2d(bins =2000,aes(fill = log1p(..count..))) 
m

n <- qplot(xlab="Longitude",ylab="Latitude",main="2014 Photos",geom="blank",x=geo_locs$V2,y=geo_locs$V1,data=geo_locs,xlim=c(-74.02,-73.95), ylim=c(40.7,40.8))  + stat_bin2d(bins =1000,aes(fill = log1p(..count..))) 
n