function x = plot_inferred(file_name, file_path)
    %clc 
    %clear all
    close all
    
    data = readtable(file_path, "ReadRowNames", false, "Delimiter", ",", "ReadVariableNames", false);
    x = data(:,[1,3,5]);
    y = data(:,[2,4,6]);
    
    x10 = 0; %-1;
    y10 = 1; % 0
    
    x20 = -x10; 
    y20 = -y10 ;  
    
    x30 = 0 ; 
    y30 = 0 ;
    
    fig = figure(2)
    
    title("Inferred path")
    xlabel('x') 
    ylabel('y')
    
    ylim([-3, 3])
    xlim([-3, 3])
    
    hold on
    grid on 
    
    plot(x{:,1},y{:,1},'r')
    plot(x{:,2},y{:,2},'g')
    plot(x{:,3},y{:,3},'b')
    plot( x10, y10, 'ro', 'MarkerFaceColor','r')  ;
    plot( x20, y20, 'go', 'MarkerFaceColor','g')  ;
    plot( x30, y30, 'bo', 'MarkerFaceColor','b')  ;
    legend('Body 1','Body 2','Body 3');

    exportgraphics(fig, strcat("plots/", file_name, ".png"), "Resolution", 300)
end 