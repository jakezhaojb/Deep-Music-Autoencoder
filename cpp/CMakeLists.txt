add_library(ae_update SHARED update.cpp)
target_link_libraries(ae_update ${CMAKE_DL_LIBS})
install(TARGETS ae_update LIBRARY DESTINATION lib)

set(BASIC sparse_autoencoder.cpp)
add_library(ae_train SHARED ae.cpp)
target_link_libraries(ae_train ${BASIC}
        "/usr/lib/libboost_filesystem.so")
install(TARGETS ae_train LIBRARY DESTINATION lib)

add_executable(ae ae_driver.cpp)
target_link_libraries(ae
    ${Boost_LIBRARIES}
    comm scheduler ae_train
    )

install(TARGETS ae RUNTIME DESTINATION bin)
