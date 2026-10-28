
function [] = Analysis(DATA,ID,TITLE,MATS,LABELS,COL,STYLE,WID,FIGURE_TYPE,WIN,TH,YSCALE,XLABEL,YLABEL,Maximal_time,Maximal_OD,inc_OD,LegendPos)

%%%%% blanking
DATA(3:end,:)=DATA(3:end,:)-repmat( (min(DATA(3:end,:)'))' ,1,size(DATA(3:end,:),2));%subtract lowest measured value from all wells
DATA(3:end,:)=DATA(3:end,:)/0.23+inc_OD;%normalize to Cuvette OD units and add starting-OD

%%%%% time
time=DATA(1,:)/3600;%convert time to hours
if isinf(Maximal_time);
else
    f=max(find(time<Maximal_time));
    DATA=DATA(:,1:f);
    time=time(1:f);
end
WIN=ceil(WIN/time(2));


%%%%% calculate growth rate (doubling time) and maximal ODs
MAXY=0;
for i=1:length(MATS)
    MAT=MATS{i};
    slope=[];
    tp=[];
    BY=[];
    for j=1:length(MAT)
        f=find(strcmp(ID,MAT{j}));
        v=DATA(f,:);
        v(find(v<TH))=NaN;
        TimePoint=length(v);
        rate_tmp=zeros(TimePoint-WIN,1);
        for k=1:(TimePoint-WIN)
            window=k:(k+WIN-1);
            if sum(isnan(v(window))); rate_tmp(k)=NaN; continue; end;
            P = polyfit(time(window),log(v(window)),1);
            rate_tmp(k)=P(1);
        end
        [slope_,tp_]=nanmax(rate_tmp);
        slope=[slope ; slope_];
        tp=[tp ; time(tp_)];
        BY=[BY ; max(v)];
        MAXY=max(MAXY,max(v));
    end
    doubling_time=round(log(2)./slope,1);
    growth_rate=slope;
    %
    doubling_time_mean = round(nanmean(doubling_time),1);
    doubling_time_ste = round(std(doubling_time)/sqrt(length(doubling_time)),1);
    growth_rate_ste=round(std(growth_rate)/sqrt(length(growth_rate)),1);
    time_start_mean = round(nanmean(tp),1);
    time_start_ste = round(std(tp)/sqrt(length(tp)),1);
    biomass_yield_mean = round(nanmean(BY),3);
    biomass_yield_ste = round(std(BY)/sqrt(length(BY)),1);
    
    LABELS{i}=[LABELS{i} ',' 'Growth rate: (h),' num2str(growth_rate) ', ±,' num2str(growth_rate_ste)...
        ',' ' at timepoint: ' num2str(time_start_mean) ', ±,' num2str(time_start_ste)...
        ',' ' max OD ,' num2str(biomass_yield_mean) ', ±, ' num2str(biomass_yield_ste)...
        ];
   disp(LABELS{i})
   
end