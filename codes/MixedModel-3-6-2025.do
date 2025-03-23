tsset cid year
mixed evi psum temp gdp gova aa gp ls iofp loh rpop hmp pte pcurei croppg grass d.urban bus_p dustgdp L.fosgdp d.invpa_p mobile_p noxgdp d.road_p telev L.wateru || cid:, mle variance nostderr
mixed evi psum temp gdp gova aa gp ls iofp loh rpop hmp pte pcurei croppg grass d.urban bus_p dustgdp L.fosgdp d.invpa_p mobile_p noxgdp d.road_p telev L.wateru || year:, mle variance nostderr

tsset cid year
mixed evi psum temp gdp gova aa gp ls iofp loh rpop hmp pte pcurei croppg grass d.urban bus_p dustgdp L.fosgdp d.invpa_p mobile_p noxgdp d.road_p telev L.wateru || cid:, vce(robust)
predict evi_p1
corr evi evi_p1
di r(rho)^2
mixed evi psum temp gdp gova aa gp ls iofp loh rpop hmp pte pcurei croppg grass d.urban bus_p dustgdp L.fosgdp d.invpa_p mobile_p noxgdp d.road_p telev L.wateru || year:, vce(robust)
predict evi_p2
corr evi evi_p2
di r(rho)^2
