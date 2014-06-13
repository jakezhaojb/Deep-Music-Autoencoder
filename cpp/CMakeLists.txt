add_library(ae_update SHARED update.cpp)
target_link_libraries(ae_update ${CMAKE_DL_LIBS})
install(TARGETS ae_update LIBRARY DESTINATION lib)

set(FILES ae.cpp)
add_library(ae_train SHARED ${FILES})
target_link_libraries(ae_train
        "/usr/lib/libboost_filesystem.so")
install(TARGETS ae_train LIBRARY DESTINATION lib)

set(FILES fine_tn.cpp)
add_library(fine_tn_train SHARED ${FILES})
target_link_libraries(fine_tn_train
        "/usr/lib/libboost_filesystem.so")
install(TARGETS fine_tn_train LIBRARY DESTINATION lib)

add_executable(ae ae_driver.cpp)
target_link_libraries(ae
    ${Boost_LIBRARIES}
    comm scheduler ae_train fine_tn_train
    )

install(TARGETS ae RUNTIME DESTINATION bin)
