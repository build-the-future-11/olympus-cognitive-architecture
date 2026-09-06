BEGIN { FS="," }
NR==1 {
  for (i=1; i<=NF; i++) idx[$i]=i
  next
}
{
  n++
  x[n]=$(idx["x"])+0
  t[n]=$(idx["treatment"])+0
  y[n]=$(idx["y"])+0
  sx+=x[n]; st+=t[n]; sy+=y[n]
}
END {
  mx=sx/n; mt=st/n; my=sy/n
  for (i=1; i<=n; i++) {
    xc=x[i]-mx; tc=t[i]-mt; yc=y[i]-my
    s11+=tc*tc; s22+=xc*xc; s12+=tc*xc
    s1y+=tc*yc; s2y+=xc*yc
  }
  det=s11*s22-s12*s12
  if (det==0) { print "singular design" > "/dev/stderr"; exit 2 }
  b1=(s1y*s22-s2y*s12)/det
  b2=(s2y*s11-s1y*s12)/det
  b0=my-b1*mt-b2*mx
  for (i=1; i<=n; i++) {
    r=y[i]-(b0+b1*t[i]+b2*x[i]); rss+=r*r
  }
  sigma2=rss/(n-3)
  se=sqrt(sigma2*s22/det)
  effect=b1
  if (ENVIRON["PANTHEON_NEGATE_METRIC"]=="1") effect=-effect
  lo=effect-1.96*se; hi=effect+1.96*se
  canary=ENVIRON["PANTHEON_CANARY_VISIBLE"]
  printf "{\n  \"effect\": %.17g,\n  \"se\": %.17g,\n  \"ci95\": [%.17g, %.17g],\n  \"n\": %d,\n  \"seed\": %d,\n  \"metric\": \"adjusted_ols_treatment_effect\"", effect,se,lo,hi,n,seed
  if (canary!="") printf ",\n  \"observed_canary\": \"%s\"", canary
  printf "\n}\n"
}
