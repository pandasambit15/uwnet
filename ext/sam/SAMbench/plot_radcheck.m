clear all; define_constants

for ncase = 1:2

clear out

  switch ncase
   case 1
    casename = '256x128_1km_10s_20160209';
    fig0 = 20;
    microname = 'THOM';
    plottag = 'KWAJEX: THOMPSON, 2D, days 223-224';
    filetag = 'KWAJEX_THOMP_SnowOnOff';
   case 2
    casename = '256x128_1km_10s_20160209';
    fig0 = 20;
    microname = 'M2005';
    plottag = 'KWAJEX: M2005, 2D, days 223-224';
    filetag = 'KWAJEX_M2005_SnowOnOff';
   otherwise 
    error('bad ncase value');
  end

n = 1;
out(n).nc = sprintf('OUT_STAT/KWAJEX_%s_MPDATA_CAM_%s.nc',casename,microname);
out(n).name = 'CAM w/Snow'; n = n + 1;
out(n).nc = sprintf('OUT_STAT/KWAJEX2_%s_NoSnowRad_MPDATA_CAM_%s.nc',casename,microname);
out(n).name = 'CAM w/o Snow'; n = n + 1;
out(n).nc = sprintf('OUT_STAT/KWAJEX3_%s_LegacyRad_MPDATA_CAM_%s.nc',casename,microname);
out(n).name = 'CAM Legacy'; n = n + 1;
out(n).nc = sprintf('OUT_STAT/KWAJEX_%s_MPDATA_RRTM_%s.nc',casename,microname);
out(n).name = 'RRTM w/Snow'; n = n + 1;
out(n).nc = sprintf('OUT_STAT/KWAJEX2_%s_NoSnowRad_MPDATA_RRTM_%s.nc',casename,microname);
out(n).name = 'RRTM w/o Snow'; n = n + 1;
out(n).nc = sprintf('OUT_STAT/KWAJEX3_%s_LegacyRad_MPDATA_RRTM_%s.nc',casename,microname);
out(n).name = 'RRTM Legacy'; n = n + 1;



frac_clb = 0.5;
disp(['Cloud base is the lowest level at which the cloud fraction'])
disp(['  achieves ' sprintf('%d',100*frac_clb) ' percent of its ' ...
      'maximum value']);

for q = 1:length(out)

  which = {'time','z','p','CWP','IWP','RWP','SWP', ...
           'LHF','SHF','PREC','CLD','CLDSHD', ...
           'SOLIN','SWNTOA','SWNTOAC','LWNTOA','LWNTOAC', ...
           'LWNS','LWNSC','SWNS','SWNSC', ...
           'RELH','U','V','W2','TL','QV','QCL','QCI','QPL','QPI', ...
           'QTFLUX','TLFLUX','BUOYA','TL2','PRECIP', ...
           'RADLWUP','RADLWDN','RADSWUP','RADSWDN', ...
           'RADQR','RADQRLW','RADQRSW','RADQRCLW','RADQRCSW','WOBS','RHO', ...
          'MODISTOT','MISRTOT','ISCCPTOT'};
  for m = 1:length(which)
    out(q).(which{m}) = nc_varget(out(q).nc,which{m});
  end
  out(q).zkm = out(q).z/1000;

  out(q).ASR = out(q).SOLIN-out(q).SWNTOA;
  out(q).SWCRE = out(q).SWNTOA - out(q).SWNTOAC;
  out(q).LWCRE = out(q).LWNTOAC - out(q).LWNTOA;
  out(q).QT = out(q).QV + out(q).QCL;
  out(q).RELH = 0.01*out(q).RELH; % convert relative humidity to a fraction

  out(q).RTOA = out(q).SWNTOA - out(q).LWNTOA;
  out(q).RSRF = out(q).SWNS - out(q).LWNS;
  out(q).RTOAC = out(q).SWNTOAC - out(q).LWNTOAC;
  out(q).RSRFC = out(q).SWNSC - out(q).LWNSC;


  % set up a grid for the w levels
  out(q).zkmw = [0; 0.5*(out(q).zkm(1:end-1)+out(q).zkm(2:end))];


  for n = 1:length(out(q).time)
    out(q).zinv(n) = NaN;
    out(q).zclb(n) = NaN;
    out(q).zbfmin(n) = NaN;
    out(q).entr(n) = NaN;
    out(q).pcp_clb(n) = NaN;

    kinv = min(find(out(q).RELH(n,:)<0.5));
    kclb = min(find(out(q).CLD(n,:)> ...
                    frac_clb*max(out(q).CLD(n,:))));
    kbfmin = min(find(out(q).BUOYA(n,1:kclb)==min(out(q).BUOYA(n,1:kclb))));

    if ~isempty(kinv)
      out(q).zinv(n) = interp1(out(q).RELH(n,kinv-1:kinv),...
                               1000*out(q).zkm(kinv-1:kinv), ...
                               0.5);
      out(q).wsub(n) = interp1(out(q).RELH(n,kinv-1:kinv),...
                               out(q).WOBS(n,kinv-1:kinv),...
                               0.5);
      if ~isnan(out(q).TL2(n,1))
        % compute inversion height, inverson base and inversion top
        %   using a method from Yamaguchi and Randall (2011, MWR, 
        %   doi:10.1175/MWR-D-10-05044.1  The inversion is taken
        %   as the location with maximum liquid static energy
        %   variance.  The inversion base and inversion top are the
        %   nearest locations where the sl variance falls to 5% of
        %   its maximum value.
        
        sl2max = max(out(q).TL2(n,:));
        k_sl2max = min(find(out(q).TL2(n,:)==sl2max));
        out(q).z_sl2max(n) = 1000*out(q).zkm(k_sl2max);

        ktop_plus = min(find(1e3*out(q).zkm>out(q).z_sl2max(n) ...
                             & out(q).TL2(n,:)' < 0.05*sl2max));
        kbot_minus = max(find(1e3*out(q).zkm<out(q).z_sl2max(n) ...
                              & out(q).TL2(n,:)' < 0.05*sl2max));
        if isempty(kbot_minus) | isempty(ktop_plus)
          out(q).z_sl2max(n) = NaN;
          out(q).zbp_sl2max(n) = NaN;
          out(q).zbm_sl2max(n) = NaN;
        else
          out(q).zbp_sl2max(n) = interp1(...
              out(q).TL2(n,k_sl2max:ktop_plus), ...
              1000*out(q).zkm(k_sl2max:ktop_plus), ...
              0.05*sl2max);
          
          out(q).zbm_sl2max(n) = interp1(...
              out(q).TL2(n,kbot_minus:k_sl2max), ...
              1000*out(q).zkm(kbot_minus:k_sl2max), ...
              0.05*sl2max);
        end

        out(q).dz_sl2max(n) = out(q).zbp_sl2max(n) - out(q).zbm_sl2max(n);
      else
        out(q).z_sl2max(n) = NaN;
      end

      % compute the bl-integrated radiative cooling
      out(q).drad(n) = quad_discretefun(1e3*out(q).zkm, ...
                                        (Cp/86400)*out(q).RHO(n,:) ...
                                        .*(out(q).RADQRLW(n,:)+out(q).RADQRSW(n,:)), ...
                                        0, out(q).zinv(n));
      out(q).drad_clr(n) = quad_discretefun(1e3*out(q).zkm, ...
                                        (Cp/86400)*out(q).RHO(n,:) ...
                                        .*(out(q).RADQRCLW(n,:)+out(q).RADQRCSW(n,:)), ...
                                        0, out(q).zinv(n));


    end

    if isempty(kclb) | kclb==1
      out(q).zclb(n) = out(q).zinv(n);
      % compute precipitation rate at cloud base.
      out(q).pcp_clb(n) = 0;
    else
      out(q).zclb(n) = interp1(out(q).CLD(n,kclb-1:kclb),...
                               1000*out(q).zkm(kclb-1:kclb), ...
                               frac_clb*max(out(q).CLD(n,:)));
      % compute precipitation rate at cloud base.
      out(q).pcp_clb(n) = interp1(out(q).z,out(q).PRECIP(n,:), ...
                                  out(q).zclb(n));
    end

    if out(q).zinv(n)>200
      qt_surf = quad_discretefun(1e3*out(q).zkm, out(q).QT(n,:), ...
                                 100, 200)/100;
      qt_cld = quad_discretefun(1e3*out(q).zkm, out(q).QT(n,:), ...
                                out(q).zinv(n)-200, out(q).zinv(n)-100)/100;
% $$$       tvl_surf = quad_discretefun(1e3*out(q).zkm, out(q).TVL(n,:), ...
% $$$                                   100, 200)/100;
% $$$       tvl_cld = quad_discretefun(1e3*out(q).zkm, out(q).TVL(n,:), ...
% $$$                                  out(q).zinv(n)-200, out(q).zinv(n)-100)/100;
      tl_surf = quad_discretefun(1e3*out(q).zkm, out(q).TL(n,:), ...
                                 100, 200)/100;
      tl_cld = quad_discretefun(1e3*out(q).zkm, out(q).TL(n,:), ...
                                out(q).zinv(n)-200, out(q).zinv(n)-100)/100;

      out(q).dtl_bl_dcpl(n) = tl_surf - tl_cld;
% $$$       out(q).dtvl_bl_dcpl(n) = tvl_surf - tvl_cld;
      out(q).dqt_bl_dcpl(n) = qt_surf - qt_cld;
      out(q).dqt_bl_dcpl_norm(n) = (qt_surf - qt_cld)/qt_surf;

      qt_mid = 0.5*(qt_surf + qt_cld);
      out(q).z_dcpl(n) = ...
          1e3*out(q).zkm(min(find(abs(out(q).QT(n,:)-qt_mid) ...
                                    == min(abs(out(q).QT(n,:)-qt_mid)))));
      
    else
% $$$       out(q).dtvl_bl_dcpl(n) = NaN;
      out(q).dqt_bl_dcpl(n) = NaN;
      out(q).dqt_bl_dcpl_norm(n) = NaN;
      out(q).z_dcpl(n) = NaN;
    end

    if ~isnan(out(q).z_sl2max(n)) 
      
      out(q).dSL_YR2011(n) = ...
          interp1(1e3*out(q).zkm, ...
                  out(q).TL(n,:)', ...
                  out(q).zbp_sl2max(n)) ...
          - interp1(1e3*out(q).zkm, ...
                    out(q).TL(n,:)', ...
                    out(q).zbm_sl2max(n));
      out(q).dQT_YR2011(n) = ...
          interp1(1e3*out(q).zkm, ...
                  1e-3*out(q).QT(n,:)', ...
                  out(q).zbp_sl2max(n)) ...
          - interp1(1e3*out(q).zkm, ...
                    1e-3*out(q).QT(n,:)', ...
                    out(q).zbm_sl2max(n));

      out(q).dFsl_YR2011(n) = ...
          interp1(1e3*out(q).zkmw, ...
                  (out(q).TLFLUX(n,:)./out(q).RHO(n,:)/Cp)', ...
                  out(q).zbp_sl2max(n)) ...
          - interp1(1e3*out(q).zkmw, ...
                    (out(q).TLFLUX(n,:)./out(q).RHO(n,:)/Cp)', ...
                    out(q).zbm_sl2max(n));

      out(q).dFrad_YR2011(n) = ...
          interp1(1e3*out(q).zkmw,...
                  (out(q).RADLWUP(n,:)' - out(q).RADLWDN(n,:)' ...
                   + out(q).RADSWUP(n,:)' - out(q).RADSWDN(n,:)') ...
                  ./(Cp*out(q).RHO(n,:)'), ...
                  out(q).zbp_sl2max(n)) ...
          - interp1(1e3*out(q).zkmw,...
                    (out(q).RADLWUP(n,:)' - out(q).RADLWDN(n,:)' ...
                     + out(q).RADSWUP(n,:)' - out(q).RADSWDN(n,:)') ...
                    ./(Cp*out(q).RHO(n,:)'), ...
                    out(q).zbm_sl2max(n));

      out(q).dFqt_YR2011(n) = ...
          interp1(1e3*out(q).zkmw, ...
                  (out(q).QTFLUX(n,:)./out(q).RHO(n,:)/L)', ...
                  out(q).zbp_sl2max(n)) ...
          - interp1(1e3*out(q).zkmw, ...
                    (out(q).QTFLUX(n,:)./out(q).RHO(n,:)/L)', ...
                    out(q).zbm_sl2max(n));

      out(q).entr_YR2011(n) = 1e3* ...
          (out(q).dSL_YR2011(n)*(out(q).dFsl_YR2011(n) + ...
                                   out(q).dFrad_YR2011(n)) ...
           + (L/Cp)^2*out(q).dQT_YR2011(n)*out(q).dFqt_YR2011(n)) ...
          ./( out(q).dSL_YR2011(n)^2 ...
              + (L/Cp*out(q).dQT_YR2011(n))^2 );

    end
  end      
end
  

stufftoplot = { ...
    {'zinv','zinv, m'}, ...
    {'zclb','cloud base, m'}, ...
    {'CWP','CWP, g m^{-2}'}, ...
    {'IWP','IWP, g m^{-2}'}, ...
    {'SWP','SWP, g m^{-2}'}, ...
    {'RWP','RWP, g m^{-2}'}, ...
    {'ISCCPTOT','ISCCP totoL cloud frac'}, ...
    {'MODISTOT','MODIS totoL cloud frac'}, ...
    {'MISRTOT','MISR totoL cloud frac'}, ...
    {'CLDSHD','Shaded cloud frac'}, ...
    {'LWCRE','LWCRE, W m^{-2}'}, ...
    {'SWCRE','SWCRE, W m^{-2}'}, ...
    {'SOLIN','SW insolation, W m^{-2}'}, ...
    {'ASR','Absorbed SW, W m^{-2}'}, ...
    {'LWNTOA','OLR, W m^{-2}'}, ...
    {'LHF','LHF, W m^{-2}'}, ...
    {'SHF','SHF, W m^{-2}'}, ...
    {'PREC','surface precip, mm d^{-1}'}, ...
    {'pcp_clb','cloud base precip, mm d^{-1}'}, ...
    {'dqt_bl_dcpl','dQT decoupling, g kg^{-1}'}, ...
    {'dtl_bl_dcpl','dSL decoupling, K'}, ...
    {'RTOA','RADFLX TOA, W m^{-2}'}, ...
    {'RTOAC','RADFLX CLR TOA, W m^{-2}'}, ...
    {'RSRF','RADFLUX SRF, W m^{-2}'}, ...
    {'RSRFC','RADFLUX CLR SRF, W m^{-2}'}, ...
    {'drad','delta RADFLUX (surf to zinv), W m^{-2}'}, ...
    {'drad_clr','delta clearsky RADFLUX (surf to zinv), W m^{-2}'}, ...
              };

%    {'drad','dRAD, W m^{-2}'}, ...
%    {'zlcl','lifting condensation level, m'}, ...
%    {'zclbmlcl','Cloud base height minus LCL, m'}, ...
%    {'drad_clr','dRAD CLR, W m^{-2}'}, ...
%    {'deltab','deltab, ms^{-2}'}, ...
%    {'wstar3','wstar3, m^3s^{-3}'}, ...
%    {'kappa','Kappa'}, ...

nfig = 1;
for kk = 1:length(stufftoplot)
  if mod(kk,3) == 1
    figure(fig0+nfig); clf
    nfig = nfig + 1;
    nsub = 1;
  else
    nsub = nsub + 1;
  end
  hLa = subplot(2,2,nsub);
  hL1 = plot_pldata('time, day',stufftoplot{kk}{2},plottag,-1,out,'time', ...
                    stufftoplot{kk}{1});
  if kk==length(stufftoplot) | mod(kk,3)==0
    nsub = nsub + 1;
    hLb = subplot(2,2,nsub);
    hL2 = plot_pldata('time, day',stufftoplot{kk}{2},'',-1,out,'time', ...
                    stufftoplot{kk}{1});
    hLL = pldata_legend(hL2,out,2);
    set(hLL,'FontSize',8);
    set(hLb,'Visible','off');
    set(hL2,'Visible','off');
  end
  eval(sprintf('print -dpng -r200 %s_diagnostic_plot%.2d.png',filetag,nfig))
end

end  