#!/usr/bin/env bash


PSQL_SERVER="dev-db-p13n-ros.dev.tripadvisor.com"
psql -U postgres -h ${PSQL_SERVER} -p 5434 p13n_ros -c "
    SELECT
        m.path,
        m.id,
        j.locationid,
        j.parentid,
        j.placetypeid
    FROM
        (SELECT
            lp.parentid,
            lp.locationid,
            l.placetypeid
        FROM
            t_location l
        JOIN
            t_locationpaths lp
        ON l.id=lp.locationid
        WHERE
            l.placetypeid=10022
            AND (lp.parentid=187849)
        ) j
    JOIN
        t_media_locations ml
    ON j.locationid=ml.locationid
    JOIN
        t_media m
    ON ml.mediaid=m.id
    " >> resources/paths/thumbnail/milan
