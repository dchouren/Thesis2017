
library(ggplot2)
geo_locs <- read.table('../2014_geo', header=FALSE, sep=',')
m <- qplot(xlab="Longitude",ylab="Latitude",main="2014 Photos",geom="blank",x=geo_locs$V2,y=geo_locs$V1,data=geo_locs,xlim=c(-74.052544,-73.740685), ylim=c(40.525070,40.889249))  + stat_bin2d(bins =2000,aes(fill = log1p(..count..))) 
m

n <- qplot(xlab="Longitude",ylab="Latitude",main="2014 Photos",geom="blank",x=geo_locs$V2,y=geo_locs$V1,data=geo_locs,xlim=c(-74.02,-73.95), ylim=c(40.7,40.8))  + stat_bin2d(bins =1000,aes(fill = log1p(..count..))) 
n
colnames(geo_locs) <- c('Long', 'Lat')

dens <- contourLines(kde2d(geo_locs$Long, geo_locs$Lat, lims=c(expand_range(range(geo_locs$Long), add=0.5), expand_range(range(geo_locs$Lat), add=0.5))))
geo_locs$Density <- 0

for (i in 1:length(dens)) {
  tmp <- point.in.polygon(geo_locs$Long, geo_locs$Lat, dens[[i]]$x, dens[[i]]$y)
  geo_locs$Density[which(tmp==1)] <- dens[[i]]$level
}

manhattan_closeup = c(-74.02, 40.69, -73.95, 40.8)
manhattan <- get_map(manhattan_closeup, zoom=12, source='google')
gg <- ggmap(manhattan, xlab='Longitude',ylab='Latitude',main='2014 Photos')
gg <- gg + geom_point(data=geo_locs, aes(x=Lat, y=Long), size=0.001, color='blue', alpha=0.01)
gg


m <- qplot(xlab='Longitude',ylab='Latitude',main='2014 Photos',geom='blank',x=Lat,y=Long,data=geo_locs,xlim=c(-74.052544,-73.70685), ylim=c(40.525070,40.889249))  + stat_bin2d(bins =2000,aes(fill = log1p(..count..))) 
m


new_york = c(-74.12, 40.50, -73.77, 40.95)
new_york = get_map(new_york, zoom=11)
new_york_map <- ggmap(new_york) + geom_point(data=geo_locs, aes(x=Lat, y=Long), size=0.0005, color='blue', alpha=0.005)
new_york_map



# manhattan = get_map(location='manhattan', zoom=12)
# map <- ggmap(manhattan,xlab='Longitude',ylab='Latitude',main='2014 Photos',geom='blank',x=geo_locs$V2,y=geo_locs$V1, xlim=c(-74.02,-73.95), ylim=c(40.69,40.8)) + stat_bin2d(bins =2000,aes(fill = log1p(..count..))) 

