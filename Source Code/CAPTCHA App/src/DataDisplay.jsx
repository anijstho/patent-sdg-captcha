import React, { useState, useEffect } from 'react';
import { Box, Card, Flex, Text, Heading, Button, IconButton } from '@radix-ui/themes';
import { CaretLeft, CaretRight, MagnifyingGlass, Funnel } from "@phosphor-icons/react";
import { fetchData, getPaginatedData, filterData, getColumns } from './dataService';

/**
 * DataDisplay component that shows data from the Python dataframe
 */
const DataDisplay = ({ isDarkMode }) => {
  // State for data and loading status
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Pagination state
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalRecords, setTotalRecords] = useState(0);
  
  // Filter state
  const [filterColumn, setFilterColumn] = useState('');
  const [filterValue, setFilterValue] = useState('');
  const [isFiltering, setIsFiltering] = useState(false);

  // Load initial data and columns
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Get column names
        const columnsResponse = await getColumns();
        if (columnsResponse.success) {
          setColumns(columnsResponse.columns);
        }
        
        // Get paginated data
        const response = await getPaginatedData(page, pageSize);
        if (response.success) {
          setData(response.data);
          setTotalRecords(response.total);
        } else {
          setError('Failed to load data');
        }
      } catch (err) {
        setError(`Error: ${err.message}`);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [page, pageSize]);

  // Apply filters
  const handleApplyFilter = async () => {
    if (!filterColumn || !filterValue) {
      return;
    }
    
    try {
      setIsFiltering(true);
      setLoading(true);
      setError(null);
      
      const filters = {
        [filterColumn]: filterValue
      };
      
      const response = await filterData(filters);
      if (response.success) {
        setData(response.data);
        setTotalRecords(response.total);
      } else {
        setError('Failed to apply filter');
      }
    } catch (err) {
      setError(`Error applying filter: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Reset filters
  const handleResetFilter = async () => {
    setIsFiltering(false);
    setFilterColumn('');
    setFilterValue('');
    
    try {
      setLoading(true);
      setError(null);
      
      const response = await getPaginatedData(1, pageSize);
      if (response.success) {
        setData(response.data);
        setTotalRecords(response.total);
        setPage(1);
      } else {
        setError('Failed to reset data');
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Total pages calculation
  const totalPages = Math.ceil(totalRecords / pageSize);

  return (
    <Card style={{ maxWidth: '800px', margin: '0 auto', marginTop: '20px' }}>
      <Box p="4">
        <Heading as="h2" size="5" mb="2" style={{ 
          color: isDarkMode ? '#ffffff' : 'var(--color-fg-default)',
          fontWeight: 600
        }}>
          Python DataFrame Data Explorer
        </Heading>
        <Text as="p" size="2" mb="4" style={{ 
          color: isDarkMode ? '#ffffff' : 'gray',
          fontWeight: 500
        }}>
          Interacting with testenv_updated.pkl dataframe
        </Text>

        {/* Filter controls */}
        <Box mb="4">
          <Flex gap="2" align="center" justify="between" mb="2">
            <Heading as="h3" size="3">Filters</Heading>
            <IconButton 
              variant="ghost" 
              onClick={handleResetFilter}
              disabled={!isFiltering}
              aria-label="Reset filters"
            >
              <Funnel weight="fill" />
            </IconButton>
          </Flex>
          
          <Flex gap="2" mb="2">
            <select 
              value={filterColumn}
              onChange={(e) => setFilterColumn(e.target.value)}
              style={{ 
                flex: 1, 
                padding: '8px', 
                borderRadius: '6px', 
                border: '1px solid var(--color-border-default)',
                backgroundColor: isDarkMode ? 'var(--color-canvas-subtle)' : 'white',
                color: isDarkMode ? 'white' : 'black'
              }}
            >
              <option value="">Select column</option>
              {columns.map(column => (
                <option key={column} value={column}>{column}</option>
              ))}
            </select>
            
            <input
              type="text"
              placeholder="Filter value"
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              style={{ 
                flex: 1, 
                padding: '8px', 
                borderRadius: '6px', 
                border: '1px solid var(--color-border-default)',
                backgroundColor: isDarkMode ? 'var(--color-canvas-subtle)' : 'white',
                color: isDarkMode ? 'white' : 'black'
              }}
            />
            
            <Button onClick={handleApplyFilter} disabled={!filterColumn || !filterValue}>
              <MagnifyingGlass weight="bold" />
              Apply
            </Button>
          </Flex>
        </Box>

        {/* Loading and error states */}
        {loading && <Box p="4" textAlign="center"><Text>Loading data...</Text></Box>}
        {error && <Box p="4" style={{ backgroundColor: '#ff5555', borderRadius: '6px' }}><Text style={{ color: 'white', fontWeight: 600 }}>{error}</Text></Box>}

        {/* Data table */}
        {!loading && !error && data.length > 0 && (
          <Box 
            style={{ 
              border: '1px solid var(--color-border-default)', 
              borderRadius: '6px',
              overflow: 'auto',
              maxHeight: '400px'
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {Object.keys(data[0]).map(key => (
                    <th 
                      key={key}
                      style={{ 
                        padding: '10px', 
                        textAlign: 'left', 
                        borderBottom: '2px solid var(--color-border-default)',
                        backgroundColor: isDarkMode ? 'var(--color-canvas-subtle)' : '#f5f5f5'
                      }}
                    >
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {Object.values(row).map((value, colIndex) => (
                      <td 
                        key={colIndex}
                        style={{ 
                          padding: '10px', 
                          borderBottom: '1px solid var(--color-border-default)',
                        }}
                      >
                        {value !== null && value !== undefined ? String(value) : 'N/A'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        )}

        {/* Pagination controls */}
        {!loading && !error && data.length > 0 && (
          <Flex justify="between" align="center" mt="4">
            <Button
              variant="soft"
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
            >
              <CaretLeft weight="bold" />
              Previous
            </Button>
            
            <Text>
              Page {page} of {totalPages} ({totalRecords} records)
            </Text>
            
            <Button
              variant="soft"
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
            >
              Next
              <CaretRight weight="bold" />
            </Button>
          </Flex>
        )}
        
        {/* No data message */}
        {!loading && !error && data.length === 0 && (
          <Box p="4" textAlign="center">
            <Text>No data found. Try changing your filters.</Text>
          </Box>
        )}
      </Box>
    </Card>
  );
};

export default DataDisplay;